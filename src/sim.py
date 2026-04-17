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

    def on_gripper(self, state: str) -> None:
        """``state`` is ``"open"`` or ``"closed"``."""

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
        gripper_width: float = 0.05,
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

        # Gripper mesh: a dark cuboid that spans from the flange (frame 6) to the tool
        # tip. Length comes from ||T6t translation||. We update its pose every joint
        # step so it rigidly follows the flange.
        self._gripper_handle: Any | None = None
        self._gripper_length: float = float(np.linalg.norm(self.kin.T6t[:3, 3]))
        self._gripper_width: float = float(gripper_width)
        if show_gripper and self._gripper_length > 1e-6:
            self._gripper_handle = self.server.scene.add_box(
                "/gripper",
                dimensions=(self._gripper_width, self._gripper_width, self._gripper_length),
                color=(60, 60, 70),
            )
            self._update_gripper_pose(self._current_q)

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
        if self._gripper_handle is None:
            return
        # Flange pose (T_B6) in base. The gripper cuboid is centered halfway between
        # the flange and the tool tip, oriented with the flange frame.
        T_b6 = self.kin.fk(q, include_tool=False)
        T_center = T_b6.copy()
        T_center[:3, 3] = T_b6[:3, 3] + (self._gripper_length / 2.0) * T_b6[:3, 2]
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

    def on_gripper(self, state: str) -> None:
        self._gui_gripper_label.value = state

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
