"""Rendering hooks for simulated robot runs.

A :class:`SimulationRenderer` observes joint targets, gripper state changes,
and scene-object additions from a :class:`me235b.robot.RobotController`. The
default :class:`NullRenderer` does nothing, so production / live-hardware code
pays no cost.

:class:`ViserRenderer` drives a live 3D viewer via ``viser`` + ``yourdfpy``:

- it loads a UR URDF and animates joint updates,
- it exposes :meth:`add_box` / :meth:`set_object_pose` for scene objects
  (Hanoi boxes and pads), and
- it exposes :meth:`trail_point` for drawing a whiteboard pen trail.

``viser`` and ``yourdfpy`` are dev-only dependencies, so they are imported
lazily inside :class:`ViserRenderer` to keep the rest of the package (and the
CLI) importable without them.
"""

from __future__ import annotations

import builtins
import os
import tempfile
import time
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np

from .kinematics import UR10e
from .transforms import R_to_wxyz


# The UR10e URDF's home pose (all joints = 0) differs from classical DH by a 180 degree
# rotation about the shoulder pan. Adding this offset on every update_cfg lines up my
# FK with the URDF to ~1e-11 m at joint, wrist, and tool frames.
UR10E_JOINT_OFFSET: np.ndarray = np.array([np.pi, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)


# --- Synthetic Robotiq 2F-140 URDF -------------------------------------------
#
# ``robot_descriptions`` only ships the 2F-85, but the Lab 3 setup almost
# certainly uses the 2F-140 (the 2F-85 physically can't fit the lab's 10 or
# 12 cm blocks). Rather than vendor external mesh assets we describe the
# 2F-140 with primitive cuboids that match its real external dimensions:
#
#   - overall length when closed: ~180 mm from the mounting face
#   - max finger separation: 140 mm (70 mm stroke per finger)
#   - body width: 85 mm, thickness: 60 mm
#
# The fingers are modelled as simple prismatic joints with a mimic on the
# opposite side. The real 2F-140 is a 5-bar adaptive linkage, so the fingertip
# doesn't travel purely laterally - but for visualization this is close
# enough and it has the right footprint, reach, and stroke.
ROBOTIQ_2F140_URDF_XML: str = """<?xml version="1.0"?>
<robot name="robotiq_2f_140_synthetic">
  <material name="body_dark"><color rgba="0.14 0.14 0.17 1.0"/></material>
  <material name="body_mid"><color rgba="0.25 0.25 0.27 1.0"/></material>
  <material name="finger_silver"><color rgba="0.76 0.76 0.78 1.0"/></material>
  <material name="pad_black"><color rgba="0.08 0.08 0.08 1.0"/></material>

  <!-- Mounting plate (what bolts to the UR flange) -->
  <link name="robotiq_140_base_link">
    <visual>
      <geometry><box size="0.075 0.060 0.050"/></geometry>
      <origin xyz="0 0 0.025" rpy="0 0 0"/>
      <material name="body_dark"/>
    </visual>
  </link>

  <!-- Main gripper body housing the motor and linkage -->
  <link name="robotiq_140_body">
    <visual>
      <geometry><box size="0.085 0.060 0.070"/></geometry>
      <origin xyz="0 0 0.035" rpy="0 0 0"/>
      <material name="body_mid"/>
    </visual>
  </link>
  <joint name="base_to_body" type="fixed">
    <parent link="robotiq_140_base_link"/>
    <child link="robotiq_140_body"/>
    <origin xyz="0 0 0.050" rpy="0 0 0"/>
  </joint>

  <!-- LEFT finger: prismatic along -y, stroke 0 (closed) .. 0.070 (open, 70 mm per side) -->
  <link name="left_finger">
    <visual>
      <geometry><box size="0.015 0.024 0.055"/></geometry>
      <origin xyz="0 -0.012 0.0275" rpy="0 0 0"/>
      <material name="finger_silver"/>
    </visual>
    <visual>
      <geometry><box size="0.028 0.008 0.040"/></geometry>
      <origin xyz="0 -0.004 0.075" rpy="0 0 0"/>
      <material name="pad_black"/>
    </visual>
  </link>
  <joint name="finger_joint" type="prismatic">
    <parent link="robotiq_140_body"/>
    <child link="left_finger"/>
    <origin xyz="0 0 0.070" rpy="0 0 0"/>
    <axis xyz="0 -1 0"/>
    <limit lower="0.000" upper="0.070" effort="100" velocity="0.1"/>
  </joint>

  <!-- RIGHT finger mirrors the left via a mimic joint -->
  <link name="right_finger">
    <visual>
      <geometry><box size="0.015 0.024 0.055"/></geometry>
      <origin xyz="0 0.012 0.0275" rpy="0 0 0"/>
      <material name="finger_silver"/>
    </visual>
    <visual>
      <geometry><box size="0.028 0.008 0.040"/></geometry>
      <origin xyz="0 0.004 0.075" rpy="0 0 0"/>
      <material name="pad_black"/>
    </visual>
  </link>
  <joint name="right_finger_joint" type="prismatic">
    <parent link="robotiq_140_body"/>
    <child link="right_finger"/>
    <origin xyz="0 0 0.070" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0.000" upper="0.070" effort="100" velocity="0.1"/>
    <mimic joint="finger_joint" multiplier="1" offset="0"/>
  </joint>
</robot>
"""


class GripperSpec:
    """How a particular gripper URDF maps ``open`` / ``closed`` / block width to its joint value."""

    __slots__ = ("max_stroke_m", "open_value", "closed_value")

    def __init__(self, *, max_stroke_m: float, open_value: float, closed_value: float) -> None:
        self.max_stroke_m = float(max_stroke_m)
        self.open_value = float(open_value)
        self.closed_value = float(closed_value)

    def value_for_width(self, width_m: float | None) -> float:
        """Interpolate the joint value that leaves just enough room for ``width_m``.

        ``width_m=None`` or any width greater than ``max_stroke_m`` yields the
        fully-open value (so unreachable blocks at least render sensibly).
        """
        if width_m is None:
            return self.closed_value
        frac = float(width_m) / self.max_stroke_m
        frac = max(0.0, min(1.0, frac))
        return self.closed_value + frac * (self.open_value - self.closed_value)


GRIPPER_SPECS: dict[str, GripperSpec] = {
    # robot_descriptions 2F-85: finger_joint 0 = fully open (~94 mm), 0.725 = closed (~17 mm).
    "robotiq_2f85": GripperSpec(max_stroke_m=0.085, open_value=0.0, closed_value=0.725),
    # Our synthetic 2F-140: finger_joint 0 = fully closed (tips touching), 0.070 = fully open (140 mm).
    "robotiq_2f140": GripperSpec(max_stroke_m=0.140, open_value=0.070, closed_value=0.0),
}


@contextmanager
def _utf8_open() -> Iterator[None]:
    """Temporarily force ``open`` to default to UTF-8 for text reads.

    ``xacrodoc`` reads UR xacro files with whatever the OS's preferred encoding is.
    On Windows that's ``cp1252``, which chokes on the UTF-8 glyphs the UR xacros contain.
    Wrapping the loader in this context manager sidesteps the issue cross-platform.
    """
    original = builtins.open

    def utf8_aware(*args: Any, **kwargs: Any):
        mode = kwargs.get("mode")
        if mode is None and len(args) >= 2:
            mode = args[1]
        mode = mode or "r"
        if "b" not in mode and "encoding" not in kwargs:
            kwargs["encoding"] = "utf-8"
        return original(*args, **kwargs)

    builtins.open = utf8_aware  # type: ignore[assignment]
    try:
        yield
    finally:
        builtins.open = original  # type: ignore[assignment]


class SimulationRenderer:
    """Base/no-op renderer. Methods return neutral defaults; override as needed."""

    def on_joint_step(self, q_class: np.ndarray, *, duration: float = 0.4) -> None:
        """Called whenever the controller commits to a new joint target."""

    def on_gripper(self, state: str, *, grasped_width_m: float | None = None) -> None:
        """``state`` is ``"open"`` or ``"closed"``.

        ``grasped_width_m`` (optional) is the width of the block being grabbed; if
        provided, an adaptive gripper can stop its fingers at the block's size
        instead of mashing them fully together.
        """

    def add_box(
        self,
        T: np.ndarray,
        dimensions: tuple[float, float, float],
        *,
        label: str = "box",
        color: tuple[int, int, int] = (120, 150, 220),
    ) -> str:
        return ""

    def set_object_pose(self, handle: str, T: np.ndarray) -> None:
        """Update the pose of a previously added scene object."""

    def trail_point(self, p_world: np.ndarray, *, pen_down: bool) -> None:
        """Feed a single pen-tip sample. Consecutive pen-down samples form a stroke."""

    def attach_to_end_effector(self, handle: str) -> None:
        """Make ``handle`` follow the tool frame on subsequent joint steps."""

    def detach_from_end_effector(self, handle: str) -> None:
        """Stop having ``handle`` follow the tool frame; freeze it at the current pose."""

    def close(self, *, wait: bool = False) -> None:
        """Tear the renderer down. If ``wait``, block until the user dismisses it."""


# Public alias; `renderer or NullRenderer()` reads nicely.
NullRenderer = SimulationRenderer


class ViserRenderer(SimulationRenderer):
    """Live 3D viewer backed by ``viser`` + ``yourdfpy``.

    Notes
    -----
    The viewer uses whatever home the URDF ships with; if your classical-DH
    joint zero visually differs from the URDF zero, pass a ``joint_offset``
    that gets added to every ``q_class`` before it's pushed to the URDF.
    """

    def __init__(
        self,
        kinematics: UR10e,
        *,
        robot_type: str = "ur10e",
        joint_offset: np.ndarray | None = None,
        frame_rate: float = 30.0,
        load_meshes: bool = True,
        initial_q: np.ndarray | None = None,
        camera_position: tuple[float, float, float] = (1.6, 1.6, 1.2),
        camera_look_at: tuple[float, float, float] = (0.3, 0.0, 0.1),
        show_tool_target_frame: bool = True,
        show_gripper: bool = True,
        gripper_type: str = "robotiq_2f140",
    ) -> None:
        try:
            import viser
            from robot_descriptions.loaders.yourdfpy import load_robot_description
            from viser.extras import ViserUrdf
        except ImportError as exc:
            raise RuntimeError(
                "ViserRenderer requires the dev dependencies 'viser', 'yourdfpy', and "
                "'robot_descriptions'. Install them with `uv sync --all-groups`."
            ) from exc

        self.kin = kinematics
        if joint_offset is None:
            self.joint_offset = UR10E_JOINT_OFFSET.copy() if robot_type in ("ur10", "ur10e") else np.zeros(6, dtype=float)
        else:
            self.joint_offset = np.asarray(joint_offset, dtype=float).reshape(6)
        self.frame_dt = 1.0 / float(frame_rate)

        self.server = viser.ViserServer(verbose=False)
        try:
            self.server.initial_camera.position = tuple(float(v) for v in camera_position)
            self.server.initial_camera.look_at = tuple(float(v) for v in camera_look_at)
        except AttributeError:
            pass
        self.server.scene.add_grid("/grid", width=2.0, height=2.0)
        self.server.scene.add_frame("/base", axes_length=0.15, axes_radius=0.005)

        with _utf8_open():
            try:
                urdf = load_robot_description(
                    f"{robot_type}_description",
                    load_meshes=load_meshes,
                    build_scene_graph=load_meshes,
                    load_collision_meshes=False,
                    build_collision_scene_graph=False,
                )
            except (ModuleNotFoundError, FileNotFoundError):
                # Fall back to the mesh-only UR10 if xacro processing is unavailable.
                urdf = load_robot_description(
                    "ur10_description",
                    load_meshes=load_meshes,
                    build_scene_graph=load_meshes,
                    load_collision_meshes=False,
                    build_collision_scene_graph=False,
                )
        self.viser_urdf = ViserUrdf(
            self.server,
            urdf_or_path=urdf,
            load_meshes=load_meshes,
            load_collision_meshes=False,
        )

        self._current_q = np.zeros(6, dtype=float) if initial_q is None else np.asarray(initial_q, dtype=float).reshape(6)
        self.viser_urdf.update_cfg(self._current_q + self.joint_offset)

        self._object_counter = 0
        self._object_handles: dict[str, Any] = {}
        self._object_local: dict[str, np.ndarray] = {}  # pose relative to end effector while attached
        self._attached: set[str] = set()

        self._trail_counter = 0
        self._prev_point: np.ndarray | None = None
        self._prev_pen_down: bool = False

        self._target_frame: Any | None = None
        if show_tool_target_frame:
            # A yellow coordinate frame that floats at the FK-computed tool pose. When
            # everything lines up, the arm's wrist sits right on top of it.
            self._target_frame = self.server.scene.add_frame(
                "/tool_target",
                axes_length=0.08,
                axes_radius=0.004,
                origin_radius=0.01,
            )
            self._update_target_frame(self._current_q)

        # Gripper: mount a 2F-140 (or any dependency-compatible URDF) onto the
        # flange. Falls back to a simple cuboid if the gripper URDF can't be
        # loaded (e.g. viser/yourdfpy/robot_descriptions unavailable).
        self._gripper_mount_frame: Any | None = None
        self._gripper_urdf: Any | None = None
        self._gripper_type: str = str(gripper_type)
        self._gripper_spec: GripperSpec | None = GRIPPER_SPECS.get(self._gripper_type)
        self._gripper_handle: Any | None = None
        self._gripper_fallback_length: float = float(np.linalg.norm(self.kin.T6t[:3, 3]))

        if show_gripper:
            if gripper_type:
                try:
                    gripper_urdf_obj = _load_gripper_urdf(
                        gripper_type, load_meshes=load_meshes, loader=load_robot_description
                    )
                    self._gripper_mount_frame = self.server.scene.add_frame(
                        "/gripper_mount", show_axes=False
                    )
                    self._gripper_urdf = ViserUrdf(
                        self.server,
                        urdf_or_path=gripper_urdf_obj,
                        load_meshes=load_meshes,
                        load_collision_meshes=False,
                        root_node_name="/gripper_mount",
                    )
                except Exception as exc:  # pragma: no cover - best effort fallback
                    print(f"[viser] gripper URDF '{gripper_type}' unavailable ({exc}); falling back to cuboid.")

            if self._gripper_urdf is None and self._gripper_fallback_length > 1e-6:
                self._gripper_handle = self.server.scene.add_box(
                    "/gripper",
                    dimensions=(0.05, 0.05, self._gripper_fallback_length),
                    color=(60, 60, 70),
                )

            self._update_gripper_pose(self._current_q)
            self.on_gripper("open")

        with self.server.gui.add_folder("Simulation"):
            self._gui_step_label = self.server.gui.add_text("Last step", "idle", disabled=True)
            self._gui_gripper_label = self.server.gui.add_text("Gripper", "open", disabled=True)

    def _current_tool_T(self, q: np.ndarray | None = None) -> np.ndarray:
        q = self._current_q if q is None else np.asarray(q, dtype=float).reshape(6)
        return self.kin.fk(q)

    def _update_target_frame(self, q: np.ndarray) -> None:
        if self._target_frame is None:
            return
        T = self._current_tool_T(q)
        self._target_frame.wxyz = tuple(float(v) for v in R_to_wxyz(T[:3, :3]))
        self._target_frame.position = tuple(float(v) for v in T[:3, 3])

    def _update_gripper_pose(self, q: np.ndarray) -> None:
        T_b6 = self.kin.fk(q, include_tool=False)

        if self._gripper_mount_frame is not None:
            # Robotiq URDF: move its parent mount frame so all meshes follow.
            self._gripper_mount_frame.wxyz = tuple(float(v) for v in R_to_wxyz(T_b6[:3, :3]))
            self._gripper_mount_frame.position = tuple(float(v) for v in T_b6[:3, 3])
            return

        if self._gripper_handle is not None:
            # Cuboid fallback: centered halfway between the flange and the tool tip.
            T_center = T_b6.copy()
            T_center[:3, 3] = T_b6[:3, 3] + (self._gripper_fallback_length / 2.0) * T_b6[:3, 2]
            self._gripper_handle.wxyz = tuple(float(v) for v in R_to_wxyz(T_center[:3, :3]))
            self._gripper_handle.position = tuple(float(v) for v in T_center[:3, 3])

    def on_joint_step(self, q_class: np.ndarray, *, duration: float = 0.4) -> None:
        q_target = np.asarray(q_class, dtype=float).reshape(6)
        q_start = self._current_q.copy()
        steps = max(1, int(round(duration / self.frame_dt)))

        for i in range(1, steps + 1):
            t = i / steps
            q_interp = q_start + t * (q_target - q_start)
            self.viser_urdf.update_cfg(q_interp + self.joint_offset)
            self._update_target_frame(q_interp)
            self._update_gripper_pose(q_interp)
            if self._attached:
                self._reposition_attached(q_interp)
            time.sleep(self.frame_dt)

        self._current_q = q_target.copy()
        self._gui_step_label.value = np.array2string(q_target, precision=3)

    def on_gripper(self, state: str, *, grasped_width_m: float | None = None) -> None:
        if hasattr(self, "_gui_gripper_label"):
            if state == "closed" and grasped_width_m is not None:
                self._gui_gripper_label.value = f"closed ({grasped_width_m * 1000:.0f} mm)"
            else:
                self._gui_gripper_label.value = state

        if self._gripper_urdf is None or self._gripper_spec is None:
            return

        if state == "open":
            value = self._gripper_spec.open_value
        else:
            value = self._gripper_spec.value_for_width(grasped_width_m)
        self._gripper_urdf.update_cfg(np.array([value], dtype=float))

    def add_box(
        self,
        T: np.ndarray,
        dimensions: tuple[float, float, float],
        *,
        label: str = "box",
        color: tuple[int, int, int] = (120, 150, 220),
    ) -> str:
        self._object_counter += 1
        name = f"/objects/{label}_{self._object_counter}"
        T = np.asarray(T, dtype=float)
        handle = self.server.scene.add_box(
            name,
            dimensions=tuple(float(d) for d in dimensions),
            color=tuple(int(c) for c in color),
            wxyz=R_to_wxyz(T[:3, :3]),
            position=tuple(float(v) for v in T[:3, 3]),
        )
        self._object_handles[name] = handle
        return name

    def set_object_pose(self, handle: str, T: np.ndarray) -> None:
        if handle not in self._object_handles:
            return
        T = np.asarray(T, dtype=float)
        obj = self._object_handles[handle]
        obj.wxyz = tuple(float(q) for q in R_to_wxyz(T[:3, :3]))
        obj.position = tuple(float(v) for v in T[:3, 3])

    def attach_to_end_effector(self, handle: str) -> None:
        if handle not in self._object_handles:
            return
        T_tool = self._current_tool_T()
        obj = self._object_handles[handle]
        T_obj = np.eye(4, dtype=float)
        # Reconstruct the object's current pose from viser's wxyz+position.
        wxyz = np.asarray(obj.wxyz, dtype=float)
        pos = np.asarray(obj.position, dtype=float)
        T_obj[:3, :3] = _wxyz_to_R(wxyz)
        T_obj[:3, 3] = pos
        T_tool_obj = np.linalg.inv(T_tool) @ T_obj
        self._object_local[handle] = T_tool_obj
        self._attached.add(handle)

    def detach_from_end_effector(self, handle: str) -> None:
        self._attached.discard(handle)
        self._object_local.pop(handle, None)

    def _reposition_attached(self, q: np.ndarray) -> None:
        T_tool = self._current_tool_T(q)
        for handle in self._attached:
            T_local = self._object_local.get(handle)
            if T_local is None:
                continue
            T_world = T_tool @ T_local
            self.set_object_pose(handle, T_world)

    def trail_point(self, p_world: np.ndarray, *, pen_down: bool) -> None:
        p = np.asarray(p_world, dtype=float).reshape(3)
        if self._prev_point is not None and pen_down and self._prev_pen_down:
            segment = np.array([[self._prev_point, p]], dtype=np.float32)
            colors = np.array([[[20, 20, 20], [20, 20, 20]]], dtype=np.uint8)
            self._trail_counter += 1
            self.server.scene.add_line_segments(
                f"/trail/seg_{self._trail_counter}",
                points=segment,
                colors=colors,
                line_width=2.0,
            )
        self._prev_point = p
        self._prev_pen_down = pen_down

    def close(self, *, wait: bool = False) -> None:
        if wait:
            try:
                input("[viser] Simulation complete. Press Enter to close the viewer...")
            except EOFError:
                pass


def _wxyz_to_R(wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in np.asarray(wxyz, dtype=float).reshape(4))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _load_gripper_urdf(gripper_type: str, *, load_meshes: bool, loader: Any) -> Any:
    """Load a gripper URDF by short name.

    ``gripper_type="robotiq_2f140"`` returns the synthetic prismatic-finger URDF
    (140 mm stroke, ~180 mm reach, no external mesh dependencies). Any other name
    is forwarded to ``robot_descriptions.loaders.yourdfpy.load_robot_description``
    as ``f"{gripper_type}_description"``.
    """
    import yourdfpy

    if gripper_type == "robotiq_2f140":
        # yourdfpy reads from a file path, so write the embedded XML to a temp file.
        fd, path = tempfile.mkstemp(prefix="robotiq_2f140_", suffix=".urdf")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(ROBOTIQ_2F140_URDF_XML)
            return yourdfpy.URDF.load(path, load_meshes=False, build_scene_graph=True)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    with _utf8_open():
        return loader(
            f"{gripper_type}_description",
            load_meshes=load_meshes,
            build_scene_graph=load_meshes,
            load_collision_meshes=False,
            build_collision_scene_graph=False,
        )
