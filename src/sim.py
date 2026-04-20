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
import threading
import time
import urllib.error
import urllib.request
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .kinematics import UR10e
from .transforms import R_to_wxyz


class DemoAbort(Exception):
    """Cooperative exit from :meth:`me235b.hanoi.HanoiTask.run` (Viser restart)."""


# The UR10e URDF's home pose (all joints = 0) differs from classical DH by a 180 degree
# rotation about the shoulder pan. Adding this offset on every update_cfg lines up my
# FK with the URDF to ~1e-11 m at joint, wrist, and tool frames.
UR10E_JOINT_OFFSET: np.ndarray = np.array([np.pi, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

# Lab 3 "frame 5 to camera" mount — ViserRenderer needs this independently of HanoiTask
# so it can position the on-arm camera body and simulate the live camera view.
T5C_LAB3: np.ndarray = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, -0.1016],
        [0.0, 0.0, 1.0, 0.0848],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)

# Lab 3 camera intrinsics from the PDF (Part 1 (c)). Same values ``HanoiDetector``
# uses — duplicated here so the renderer stays decoupled from the detector module.
LAB3_CAMERA_K: np.ndarray = np.array(
    [[1698.75, 0.0, 1115.55], [0.0, 1695.98, 751.98], [0.0, 0.0, 1.0]],
    dtype=float,
)
LAB3_CAMERA_D: np.ndarray = np.array(
    [-0.00670872, -0.1481124, -0.00250596, 0.00299921, -1.68711031],
    dtype=float,
)


def _lab3_scaled_pinhole_bundle(
    panel_w: int, *, use_distortion: bool
) -> tuple[tuple[int, int], np.ndarray, np.ndarray]:
    """Return ``((W, H), K, D)`` for a Lab-3 style virtual image of width ``panel_w``."""
    native_w = float(2 * LAB3_CAMERA_K[0, 2])
    native_h = float(2 * LAB3_CAMERA_K[1, 2])
    panel_h = int(round(int(panel_w) * native_h / native_w))
    s = float(panel_w) / native_w
    K = LAB3_CAMERA_K.copy()
    K[0, 0] *= s
    K[1, 1] *= s
    K[0, 2] *= s
    K[1, 2] *= s
    D = LAB3_CAMERA_D.copy() if use_distortion else np.zeros(5, dtype=float)
    return (int(panel_w), panel_h), K, D


# --- ROS-Industrial Robotiq URDF fetcher -------------------------------------
#
# ``robot_descriptions`` only ships the 2F-85. The Lab 3 setup uses the 2F-140
# (larger stroke, longer fingers). We fetch the canonical ROS-Industrial
# `robotiq_2f_140_gripper_visualization` xacros and meshes from the
# `ros-industrial-attic/robotiq` repo on first use and cache them under the
# user's home cache directory. The URDF is produced by running the top-level
# xacro through ``xacrodoc`` with the cache dir registered as a ROS package,
# then loaded via ``yourdfpy`` with a filename handler that maps `package://`
# URIs to our cache layout.
ROSI_ROBOTIQ_BASE_URL: str = (
    "https://raw.githubusercontent.com/ros-industrial-attic/robotiq/"
    "kinetic-devel/robotiq_2f_140_gripper_visualization"
)

# Files we need pulled from the repo. Kept in a constant so the first-run
# bootstrap is transparent and easy to audit.
ROSI_ROBOTIQ_140_FILES: tuple[str, ...] = (
    "urdf/robotiq_arg2f.xacro",
    "urdf/robotiq_arg2f_140_model.xacro",
    "urdf/robotiq_arg2f_140_model_macro.xacro",
    "urdf/robotiq_arg2f_transmission.xacro",
    "meshes/visual/robotiq_arg2f_140_inner_finger.stl",
    "meshes/visual/robotiq_arg2f_140_inner_knuckle.stl",
    "meshes/visual/robotiq_arg2f_140_outer_finger.stl",
    "meshes/visual/robotiq_arg2f_140_outer_knuckle.stl",
    "meshes/visual/robotiq_arg2f_base_link.stl",
    "meshes/visual/robotiq_arg2f_coupling.stl",
    "meshes/collision/robotiq_arg2f_140_inner_finger.stl",
    "meshes/collision/robotiq_arg2f_140_inner_knuckle.stl",
    "meshes/collision/robotiq_arg2f_140_outer_finger.stl",
    "meshes/collision/robotiq_arg2f_140_outer_knuckle.stl",
    "meshes/collision/robotiq_arg2f_base_link.stl",
    "meshes/collision/robotiq_arg2f_coupling.stl",
)


def _me235b_cache_dir() -> Path:
    """Per-user cache directory for downloaded URDF assets."""
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    root = base / "me235b"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _fetch_rosi_robotiq_140(force: bool = False, verbose: bool = True) -> Path:
    """Download the ROS-Industrial 2F-140 package into the cache. Returns its package root.

    Idempotent: skips files that already exist with the expected size.
    """
    pkg_dir = _me235b_cache_dir() / "robotiq_2f_140_gripper_visualization"
    for rel in ROSI_ROBOTIQ_140_FILES:
        dst = pkg_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not force and dst.exists() and dst.stat().st_size > 0:
            continue
        url = f"{ROSI_ROBOTIQ_BASE_URL}/{rel}"
        if verbose:
            print(f"[sim] downloading {rel}...")
        tmp = dst.with_suffix(dst.suffix + ".part")
        try:
            urllib.request.urlretrieve(url, str(tmp))
            tmp.replace(dst)
        except urllib.error.URLError as exc:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to download {url}: {exc}") from exc

    # A minimal ROS package.xml so xacrodoc / rospkg will locate the package
    # via ROS_PACKAGE_PATH = cache_dir/.
    pkg_xml = pkg_dir / "package.xml"
    if not pkg_xml.exists():
        pkg_xml.write_text(
            '<?xml version="1.0"?>\n'
            '<package format="2">\n'
            '  <name>robotiq_2f_140_gripper_visualization</name>\n'
            '  <version>0.0.1</version>\n'
            '  <description>Cached subset from ros-industrial-attic/robotiq for offline '
            'rendering.</description>\n'
            '  <maintainer email="none@example.com">cache</maintainer>\n'
            '  <license>BSD</license>\n'
            '</package>\n',
            encoding="utf-8",
        )
    return pkg_dir


def _process_rosi_xacro(pkg_dir: Path) -> str:
    """Run xacrodoc on the top-level 2F-140 xacro and return the resulting URDF XML string."""
    from xacrodoc import XacroDoc, packages

    top = pkg_dir / "urdf" / "robotiq_arg2f_140_model.xacro"

    # Tell xacrodoc's own package finder (separate from rospkg) where the
    # downloaded package lives, so `$(find robotiq_2f_140_gripper_visualization)`
    # resolves to our cache directory.
    packages.update_package_cache({pkg_dir.name: str(pkg_dir)})

    with _utf8_open():
        doc = XacroDoc.from_file(str(top), walk_up=False)
        return doc.to_urdf_string()


def _load_rosi_robotiq_140(load_meshes: bool = True) -> Any:
    """Ensure the assets are cached, process the xacro, load via yourdfpy."""
    import yourdfpy

    pkg_dir = _fetch_rosi_robotiq_140()
    urdf_xml = _process_rosi_xacro(pkg_dir)

    # xacrodoc pre-resolves `package://robotiq_2f_140_gripper_visualization/...`
    # into `file://<absolute path>` before we ever see it. On Windows those look
    # like `file://C:\Users\...\mesh.stl` with backslashes, which trimesh can't
    # parse. Strip the scheme so yourdfpy just gets a plain local path.
    pkg_name = pkg_dir.name
    package_prefix = f"package://{pkg_name}/"

    def filename_handler(fname: str) -> str:
        if fname.startswith("file://"):
            return fname[len("file://"):]
        if fname.startswith(package_prefix):
            return str(pkg_dir / fname[len(package_prefix):])
        return fname

    tmp = tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False, encoding="utf-8")
    try:
        tmp.write(urdf_xml)
        tmp.close()
        return yourdfpy.URDF.load(
            tmp.name,
            build_scene_graph=True,
            load_meshes=load_meshes,
            build_collision_scene_graph=False,
            load_collision_meshes=False,
            filename_handler=filename_handler,
        )
    finally:
        Path(tmp.name).unlink(missing_ok=True)


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
        fully-closed value (so unreachable blocks at least render sensibly).
        """
        if width_m is None:
            return self.closed_value
        frac = float(width_m) / self.max_stroke_m
        frac = max(0.0, min(1.0, frac))
        return self.closed_value + frac * (self.open_value - self.closed_value)


GRIPPER_SPECS: dict[str, GripperSpec] = {
    # robot_descriptions 2F-85 and the ROS-Industrial 2F-140 share the same
    # revolute-joint convention: finger_joint = 0 is fully open, the upper
    # limit is fully closed.
    "robotiq_2f85": GripperSpec(max_stroke_m=0.085, open_value=0.0, closed_value=0.725),
    "robotiq_2f140": GripperSpec(max_stroke_m=0.140, open_value=0.0, closed_value=0.7),
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

    def add_aruco_tag(
        self,
        marker_id: int,
        T: np.ndarray,
        size_m: float,
        *,
        parent: str = "",
        dictionary: int | None = None,
    ) -> str:
        """Render an ArUco marker as a flat image quad.

        ``T`` is the marker pose; if ``parent`` is a non-empty scene path, the
        marker is added as a child of that object and ``T`` is interpreted in
        the parent's local frame (so the tag moves rigidly with the parent).
        ``dictionary`` is an ``cv2.aruco.DICT_*`` constant; default matches the
        Lab-3 dictionary.
        """
        return ""

    def set_object_pose(self, handle: str, T: np.ndarray) -> None:
        """Update the pose of a previously added scene object."""

    def trail_point(self, p_world: np.ndarray, *, pen_down: bool) -> None:
        """Feed a single pen-tip sample. Consecutive pen-down samples form a stroke."""

    def set_camera_scan_path(self, points_xyz: np.ndarray | None, *, visible: bool = True) -> None:
        """Optional base-frame polyline through planned camera scan positions."""

    def set_tag_workspace_outline(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        *,
        z_table: float = 0.002,
        visible: bool = True,
    ) -> None:
        """Optional rectangle on the table for the Lab-3 tag / tower workspace."""

    def on_scan_snapshot(
        self,
        index: int,
        *,
        total: int,
        q: np.ndarray | None = None,
        frame_bgr: np.ndarray | None = None,
    ) -> None:
        """Record one of ``total`` scan images at pose ``q`` or from a live ``frame_bgr``."""

    def synthetic_camera_bgr_for_pose(self, q: np.ndarray) -> np.ndarray | None:
        """BGR pinhole view at lab ``K`` / resolution for ArUco, or ``None``."""

    def clear_scan_pose_estimates(self) -> None:
        """Remove transient scan-time pose markers from the scene."""

    def clear_hanoi_scene_objects(self) -> None:
        """Remove pads/boxes/tags added for Hanoi so the scene can be repopulated."""

    def add_scan_pose_estimate(self, name: str, T: np.ndarray) -> None:
        """Draw a small world-frame triad at an estimated tag pose (e.g. during scan)."""

    def on_demo_checkpoint(self) -> None:
        """Interactive Viser demo: honor pause / restart between motions."""

    def enable_interactive_sim_demo(self, task: Any) -> None:
        """Install Randomize / Start / Pause / Resume / Restart controls (Viser + sim only)."""

    def wait_for_sim_demo_start(self) -> None:
        """Block until the user clicks **Start** (no-op if interactive demo is off)."""

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
        show_arm_camera: bool = True,
        show_camera_scan_path: bool = True,
        show_tag_workspace_outline: bool = True,
        show_scan_gallery: bool = True,
        scan_gallery_thumb_width: int = 200,
        show_camera_view: bool = True,
        camera_view_size: tuple[int, int] = (320, 240),
        camera_view_distortion: bool = False,
        T5c: np.ndarray | None = None,
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
        self.T5c = T5C_LAB3.copy() if T5c is None else np.asarray(T5c, dtype=float)

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

        # World-pose bookkeeping for the simulated camera view. We track top-level
        # object poses here (updated in set_object_pose / attach_to_end_effector).
        # Child primitives (ArUco tags) compose their parent's world pose with
        # their own recorded local offset at render time.
        self._world_poses: dict[str, np.ndarray] = {}
        self._primitives: list[dict[str, Any]] = []

        self._trail_counter = 0
        self._prev_point: np.ndarray | None = None
        self._prev_pen_down: bool = False
        self._show_camera_scan_path: bool = bool(show_camera_scan_path)
        self._scan_path_names: list[str] = []
        self._show_tag_workspace_outline: bool = bool(show_tag_workspace_outline)
        self._workspace_outline_names: list[str] = []
        self._scan_pose_names: list[str] = []

        self._demo_interactive: bool = False
        self._sim_paused: bool = False
        self._sim_abort_requested: bool = False
        self._sim_start_event = threading.Event()
        self._gui_demo_status: Any | None = None
        self._btn_randomize: Any | None = None
        self._btn_start: Any | None = None
        self._btn_pause: Any | None = None
        self._btn_resume: Any | None = None
        self._btn_restart: Any | None = None

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

        # Arm-mounted camera body: a dark cuboid parented to a frame that
        # tracks frame 5, with the body offset by the T5c origin so the visual
        # sits exactly where the physical camera is bolted on. We add a small
        # lens cylinder-ish cuboid too for visual clarity.
        self._camera_arm_frame: Any | None = None
        if show_arm_camera:
            self._camera_arm_frame = self.server.scene.add_frame(
                "/camera_mount", show_axes=False
            )
            camera_body_pos = tuple(float(v) for v in self.T5c[:3, 3])
            self.server.scene.add_box(
                "/camera_mount/body",
                dimensions=(0.04, 0.025, 0.030),
                color=(25, 25, 30),
                wxyz=R_to_wxyz(self.T5c[:3, :3]),
                position=camera_body_pos,
            )
            lens_offset = np.asarray(self.T5c[:3, 3], dtype=float) + self.T5c[:3, :3] @ np.array([0.0, 0.0, 0.018])
            self.server.scene.add_box(
                "/camera_mount/lens",
                dimensions=(0.016, 0.016, 0.008),
                color=(70, 70, 90),
                wxyz=R_to_wxyz(self.T5c[:3, :3]),
                position=tuple(float(v) for v in lens_offset),
            )
            self._update_camera_mount_pose(self._current_q)

        # Live camera panel + per-scan thumbnails share the same Lab-3 intrinsics
        # scaling; gallery can exist without the main floating preview.
        self._camera_view_size: tuple[int, int] = (0, 0)
        self._camera_view_K: np.ndarray | None = None
        self._camera_view_D: np.ndarray | None = None
        self._gui_camera_image: Any | None = None
        if show_camera_view:
            pw = int(camera_view_size[0])
            self._camera_view_size, self._camera_view_K, self._camera_view_D = _lab3_scaled_pinhole_bundle(
                pw, use_distortion=camera_view_distortion
            )
            w, h = self._camera_view_size
            placeholder = np.full((h, w, 3), 60, dtype=np.uint8)
            self._gui_camera_image = self.server.gui.add_image(
                placeholder,
                label="Camera view (arm)",
            )

        self._show_scan_gallery: bool = bool(show_scan_gallery)
        self._scan_gallery_size: tuple[int, int] = (0, 0)
        self._scan_gallery_K: np.ndarray | None = None
        self._scan_gallery_D: np.ndarray | None = None
        self._gui_scan_images: list[Any] = []
        self._scan_gallery_slots: int = 0
        if self._show_scan_gallery:
            tw = int(scan_gallery_thumb_width)
            # Pinhole only: Lab-3 k3 is huge; projectPoints + polygon fills and
            # tag warpPerspective assume a projective camera, so distortion here
            # bends quads into extreme, unrealistic shapes.
            self._scan_gallery_size, self._scan_gallery_K, self._scan_gallery_D = _lab3_scaled_pinhole_bundle(
                tw, use_distortion=False
            )

        # Full Lab-3 pixel size for ArUco / previews; still pinhole (D=0) for the
        # same reason as the scan gallery — see comment above.
        det_w = int(round(2.0 * float(LAB3_CAMERA_K[0, 2])))
        self._detection_size_wh, self._detection_K, self._detection_D = _lab3_scaled_pinhole_bundle(
            det_w, use_distortion=False
        )

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

    def _update_camera_mount_pose(self, q: np.ndarray) -> None:
        if self._camera_arm_frame is None:
            return
        T_b5 = self.kin.fk_to_frame(q, 5)
        self._camera_arm_frame.wxyz = tuple(float(v) for v in R_to_wxyz(T_b5[:3, :3]))
        self._camera_arm_frame.position = tuple(float(v) for v in T_b5[:3, 3])

    def _primitive_world_pose(self, prim: dict[str, Any]) -> np.ndarray | None:
        """World pose of a scene primitive; composes parent + local for tags."""
        parent = prim["parent"]
        if parent:
            base = self._world_poses.get(parent)
            if base is None:
                return None
            return base @ prim["local_T"]
        return self._world_poses.get(prim["path"])

    def _render_camera_view(self, q: np.ndarray) -> None:
        if self._gui_camera_image is None or self._camera_view_K is None:
            return
        img = self._render_camera_pixels(q, self._camera_view_size, self._camera_view_K, self._camera_view_D)
        self._gui_camera_image.image = img

    def _render_camera_pixels(
        self,
        q: np.ndarray,
        size_wh: tuple[int, int],
        K: np.ndarray,
        D: np.ndarray,
    ) -> np.ndarray:
        """Simulated on-arm image (``H×W×3`` ``uint8``) for joint vector ``q``."""
        import cv2

        q = np.asarray(q, dtype=float).reshape(6)
        W, H = int(size_wh[0]), int(size_wh[1])

        T_base_cam = self.kin.fk_to_frame(q, 5) @ self.T5c
        T_cam_base = np.linalg.inv(T_base_cam)
        R_cw = T_cam_base[:3, :3]
        t_cw = T_cam_base[:3, 3]
        rvec, _ = cv2.Rodrigues(R_cw)
        tvec = t_cw.reshape(3, 1)

        img = np.full((H, W, 3), 40, dtype=np.uint8)

        drawables: list[tuple[float, str, dict[str, Any], np.ndarray]] = []
        for prim in self._primitives:
            T_world = self._primitive_world_pose(prim)
            if T_world is None:
                continue
            origin_cam = R_cw @ T_world[:3, 3] + t_cw
            if origin_cam[2] <= 0.01:
                continue
            drawables.append((float(origin_cam[2]), prim["kind"], prim, T_world))

        drawables.sort(key=lambda x: -x[0])

        for _z, kind, prim, T_world in drawables:
            try:
                if kind == "box":
                    self._draw_box_primitive(img, prim, T_world, rvec, tvec, K, D)
                elif kind == "tag":
                    self._draw_tag_primitive(img, prim, T_world, rvec, tvec, K, D)
            except Exception:
                continue

        return img

    def synthetic_camera_bgr_for_pose(self, q: np.ndarray) -> np.ndarray | None:
        """Lab-3 detector resolution, pinhole projection, **BGR** (OpenCV order)."""
        import cv2

        rgb = self._render_camera_pixels(
            q, self._detection_size_wh, self._detection_K, self._detection_D
        )
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def clear_scan_pose_estimates(self) -> None:
        for name in self._scan_pose_names:
            try:
                self.server.scene.remove_by_name(name)
            except Exception:
                pass
        self._scan_pose_names.clear()

    def add_scan_pose_estimate(self, name: str, T: np.ndarray) -> None:
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)[:120]
        path = f"/scan_pose_est/{safe}"
        try:
            self.server.scene.remove_by_name(path)
        except Exception:
            pass
        Tm = np.asarray(T, dtype=float)
        fr = self.server.scene.add_frame(
            path,
            axes_length=0.038,
            axes_radius=0.0025,
            origin_radius=0.006,
        )
        fr.wxyz = tuple(float(v) for v in R_to_wxyz(Tm[:3, :3]))
        fr.position = tuple(float(v) for v in Tm[:3, 3])
        self._scan_pose_names.append(path)

    def on_demo_checkpoint(self) -> None:
        if not self._demo_interactive:
            return
        while self._sim_paused:
            time.sleep(0.05)
        if self._sim_abort_requested:
            self._sim_abort_requested = False
            raise DemoAbort()

    def wait_for_sim_demo_start(self) -> None:
        if not self._demo_interactive:
            return
        self._sim_start_event.wait()
        self._sim_start_event.clear()

    def enable_interactive_sim_demo(self, task: Any) -> None:
        self._demo_interactive = True
        wtask = weakref.ref(task)

        def _set_status(msg: str) -> None:
            if self._gui_demo_status is not None:
                self._gui_demo_status.value = msg

        with self.server.gui.add_folder("Lab 3 sim demo"):
            self._gui_demo_status = self.server.gui.add_text("Status", "Use Randomize, then Start", disabled=True)
            self._btn_randomize = self.server.gui.add_button("Randomize layout")

            def _on_randomize(_):
                t = wtask()
                if t is None:
                    return
                t.randomize_sim_layout()
                _set_status("Layout updated. Click Start when ready.")

            self._btn_randomize.on_click(_on_randomize)

            self._btn_start = self.server.gui.add_button("Start", color=(80, 160, 90))

            def _on_start(_):
                self._sim_start_event.set()
                _set_status("Running…")

            self._btn_start.on_click(_on_start)

            self._btn_pause = self.server.gui.add_button("Pause")

            def _on_pause(_):
                self._sim_paused = True
                _set_status("Paused")

            self._btn_pause.on_click(_on_pause)

            self._btn_resume = self.server.gui.add_button("Resume")

            def _on_resume(_):
                self._sim_paused = False
                _set_status("Running…")

            self._btn_resume.on_click(_on_resume)

            self._btn_restart = self.server.gui.add_button("Restart demo", color=(200, 120, 60))

            def _on_restart(_):
                self._sim_abort_requested = True
                self._sim_paused = False
                _set_status("Restart requested…")

            self._btn_restart.on_click(_on_restart)

        t0 = wtask()
        if t0 is not None:
            t0.randomize_sim_layout()
            _set_status("Randomized once. Use Randomize / Start.")

    def _project(self, points_world: np.ndarray, rvec, tvec, K, D) -> np.ndarray:
        import cv2

        img_pts, _ = cv2.projectPoints(points_world.astype(np.float64), rvec, tvec, K, D)
        return img_pts.reshape(-1, 2)

    def _draw_box_primitive(
        self,
        img: np.ndarray,
        prim: dict[str, Any],
        T_world: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        K: np.ndarray,
        D: np.ndarray,
    ) -> None:
        import cv2

        dx, dy, dz = prim["dims"]
        r, g, b = prim["color"]
        # 8 corners in the box's local frame.
        sx, sy, sz = dx / 2, dy / 2, dz / 2
        local_corners = np.array(
            [
                [-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz],
                [-sx, -sy, sz],  [sx, -sy, sz],  [sx, sy, sz],  [-sx, sy, sz],
            ],
            dtype=float,
        )
        world_corners = (T_world[:3, :3] @ local_corners.T).T + T_world[:3, 3]

        # Skip rendering if any corner is behind the camera (projectPoints returns
        # NaN/absurd values in that case and the integer cast blows up).
        R_cw_full = cv2.Rodrigues(rvec)[0]
        corners_cam_z = (R_cw_full @ world_corners.T).T[:, 2] + tvec.reshape(3)[2]
        if np.any(corners_cam_z <= 0.01):
            return
        corners_2d = self._project(world_corners, rvec, tvec, K, D)
        if not np.all(np.isfinite(corners_2d)):
            return

        # 6 faces, each as 4 indices into world_corners (CCW viewed from outside).
        faces = [
            (0, 3, 2, 1),  # -z
            (4, 5, 6, 7),  # +z
            (0, 1, 5, 4),  # -y
            (2, 3, 7, 6),  # +y
            (1, 2, 6, 5),  # +x
            (0, 4, 7, 3),  # -x
        ]

        R_cw = cv2.Rodrigues(rvec)[0]
        cam_origin_world = -R_cw.T @ tvec.reshape(3)

        # Sort faces by depth descending (paint far first).
        def face_depth(f):
            center = world_corners[list(f)].mean(axis=0)
            return -float(np.linalg.norm(center - cam_origin_world))

        for face in sorted(faces, key=face_depth):
            centroid = world_corners[list(face)].mean(axis=0)
            # Face normal in world = cross of (p1-p0) x (p3-p0) (approx, CCW order).
            p0, p1, p2, p3 = [world_corners[i] for i in face]
            normal = np.cross(p1 - p0, p3 - p0)
            n = np.linalg.norm(normal)
            if n > 1e-9:
                normal /= n
                # If normal points away from camera, skip (back-face cull).
                if np.dot(normal, centroid - cam_origin_world) >= 0:
                    continue

            # cv2.projectPoints under distortion can produce large or NaN
            # values for points near the optical axis - clip before casting.
            face_pts = np.array([corners_2d[i] for i in face], dtype=float)
            if not np.all(np.isfinite(face_pts)):
                continue
            face_pts = np.clip(face_pts, -32000, 32000)
            pts = face_pts.astype(np.int32)
            # Very lambertian-ish shading: darken faces whose normal is nearly parallel to view.
            if n > 1e-9:
                view_dir = (cam_origin_world - centroid) / max(np.linalg.norm(cam_origin_world - centroid), 1e-6)
                brightness = 0.6 + 0.4 * float(max(0.0, np.dot(normal, view_dir)))
            else:
                brightness = 1.0
            # viser GUI images are RGB; our prim["color"] is (r, g, b) so we
            # write the triple in RGB order. (Earlier code had a B/R swap bug.)
            fill = (
                int(np.clip(r * brightness, 0, 255)),
                int(np.clip(g * brightness, 0, 255)),
                int(np.clip(b * brightness, 0, 255)),
            )
            cv2.fillConvexPoly(img, pts, fill)
            cv2.polylines(img, [pts], True, (10, 10, 10), 1, cv2.LINE_AA)

    def _draw_tag_primitive(
        self,
        img: np.ndarray,
        prim: dict[str, Any],
        T_world: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        K: np.ndarray,
        D: np.ndarray,
    ) -> None:
        import cv2

        size = prim["size_m"]
        s = size / 2
        local_corners = np.array(
            [[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]],
            dtype=float,
        )
        world_corners = (T_world[:3, :3] @ local_corners.T).T + T_world[:3, 3]
        corners_2d = self._project(world_corners, rvec, tvec, K, D).astype(np.float32)
        if not np.all(np.isfinite(corners_2d)):
            return

        tag_img = prim["image"]
        Hs = tag_img.shape[0]
        Ws = tag_img.shape[1]
        src = np.array(
            [[0, 0], [Ws - 1, 0], [Ws - 1, Hs - 1], [0, Hs - 1]],
            dtype=np.float32,
        )
        try:
            M = cv2.getPerspectiveTransform(src, corners_2d)
        except cv2.error:
            return

        H_img, W_img = img.shape[:2]
        warped = cv2.warpPerspective(tag_img, M, (W_img, H_img))
        # Use the warped image where it's non-zero (tag pixels).
        mask = cv2.warpPerspective(
            np.full((Hs, Ws), 255, dtype=np.uint8), M, (W_img, H_img)
        )
        mask3 = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0
        img[:] = (mask3 * warped + (1 - mask3) * img).astype(np.uint8)

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
            self._update_camera_mount_pose(q_interp)
            if self._attached:
                self._reposition_attached(q_interp)
            self._render_camera_view(q_interp)
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
        dims = tuple(float(d) for d in dimensions)
        col = tuple(int(c) for c in color)
        handle = self.server.scene.add_box(
            name,
            dimensions=dims,
            color=col,
            wxyz=R_to_wxyz(T[:3, :3]),
            position=tuple(float(v) for v in T[:3, 3]),
        )
        self._object_handles[name] = handle
        self._world_poses[name] = T.copy()
        self._primitives.append(
            {"kind": "box", "path": name, "parent": "", "local_T": np.eye(4), "dims": dims, "color": col}
        )
        return name

    def clear_hanoi_scene_objects(self) -> None:
        self._attached.clear()
        self._object_local.clear()
        for path in sorted(self._object_handles.keys(), key=lambda p: (-p.count("/"), -len(p))):
            try:
                self.server.scene.remove_by_name(path)
            except Exception:
                pass
        self._object_handles.clear()
        self._world_poses.clear()
        self._primitives.clear()
        self._object_counter = 0

    def set_object_pose(self, handle: str, T: np.ndarray) -> None:
        if handle not in self._object_handles:
            return
        T = np.asarray(T, dtype=float)
        obj = self._object_handles[handle]
        obj.wxyz = tuple(float(q) for q in R_to_wxyz(T[:3, :3]))
        obj.position = tuple(float(v) for v in T[:3, 3])
        self._world_poses[handle] = T.copy()

    def add_aruco_tag(
        self,
        marker_id: int,
        T: np.ndarray,
        size_m: float,
        *,
        parent: str = "",
        dictionary: int | None = None,
    ) -> str:
        try:
            img = _generate_aruco_image(int(marker_id), dictionary=dictionary)
        except Exception as exc:  # pragma: no cover - best-effort visual
            print(f"[viser] ArUco id={marker_id} render skipped ({exc})")
            return ""

        if parent:
            name = f"{parent}/aruco"
        else:
            self._object_counter += 1
            name = f"/objects/aruco_{marker_id}_{self._object_counter}"

        T = np.asarray(T, dtype=float)
        handle = self.server.scene.add_image(
            name,
            image=img,
            render_width=float(size_m),
            render_height=float(size_m),
            wxyz=R_to_wxyz(T[:3, :3]),
            position=tuple(float(v) for v in T[:3, 3]),
        )
        self._object_handles[name] = handle
        if parent:
            # Child tag: record local offset so we can compose world pose at render time.
            self._primitives.append(
                {"kind": "tag", "path": name, "parent": parent, "local_T": T.copy(), "size_m": float(size_m), "image": img}
            )
        else:
            self._world_poses[name] = T.copy()
            self._primitives.append(
                {"kind": "tag", "path": name, "parent": "", "local_T": np.eye(4), "size_m": float(size_m), "image": img}
            )
        return name

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

    def set_camera_scan_path(self, points_xyz: np.ndarray | None, *, visible: bool = True) -> None:
        for name in self._scan_path_names:
            try:
                self.server.scene.remove_by_name(name)
            except Exception:
                pass
        self._scan_path_names.clear()
        if not self._show_camera_scan_path or points_xyz is None or not visible:
            return
        pts = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
        if pts.shape[0] < 2:
            return
        rgb = np.array([220, 120, 40], dtype=np.uint8)
        n_seg = pts.shape[0] - 1
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        colors = np.broadcast_to(rgb, (n_seg, 2, 3)).copy()
        line_name = "/scan_path/loop"
        self.server.scene.add_line_segments(
            line_name,
            points=segs,
            colors=colors,
            line_width=3.0,
        )
        self._scan_path_names.append(line_name)
        for i, row in enumerate(pts):
            sph = f"/scan_path/p_{i}"
            self.server.scene.add_icosphere(
                sph,
                radius=0.014,
                color=(220, 120, 40),
                subdivisions=2,
                position=tuple(float(v) for v in row),
            )
            self._scan_path_names.append(sph)

    def set_tag_workspace_outline(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        *,
        z_table: float = 0.002,
        visible: bool = True,
    ) -> None:
        for name in self._workspace_outline_names:
            try:
                self.server.scene.remove_by_name(name)
            except Exception:
                pass
        self._workspace_outline_names.clear()
        if not self._show_tag_workspace_outline or not visible:
            return
        z = float(z_table)
        xa, xb = float(x_min), float(x_max)
        ya, yb = float(y_min), float(y_max)
        corners = np.array(
            [
                [xa, ya, z],
                [xb, ya, z],
                [xb, yb, z],
                [xa, yb, z],
                [xa, ya, z],
            ],
            dtype=np.float32,
        )
        segs = np.stack([corners[:-1], corners[1:]], axis=1)
        rgb = np.array([90, 200, 120], dtype=np.uint8)
        colors = np.broadcast_to(rgb, (segs.shape[0], 2, 3)).copy()
        name = "/lab3_workspace/outline"
        self.server.scene.add_line_segments(
            name,
            points=segs,
            colors=colors,
            line_width=2.5,
        )
        self._workspace_outline_names.append(name)

    def on_scan_snapshot(
        self,
        index: int,
        *,
        total: int,
        q: np.ndarray | None = None,
        frame_bgr: np.ndarray | None = None,
    ) -> None:
        if not self._show_scan_gallery or self._scan_gallery_K is None:
            return
        import cv2

        if total <= 0 or index < 0 or index >= total:
            return

        if self._scan_gallery_slots != total:
            self._gui_scan_images.clear()
            with self.server.gui.add_folder("Scan views"):
                wg, hg = self._scan_gallery_size
                for i in range(total):
                    blank = np.full((hg, wg, 3), 48, dtype=np.uint8)
                    self._gui_scan_images.append(
                        self.server.gui.add_image(blank, label=f"Scan {i + 1} / {total}")
                    )
            self._scan_gallery_slots = total

        wg, hg = self._scan_gallery_size
        if frame_bgr is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = cv2.resize(rgb, (wg, hg), interpolation=cv2.INTER_AREA)
        else:
            q_use = self._current_q if q is None else np.asarray(q, dtype=float).reshape(6)
            img = self._render_camera_pixels(
                q_use, self._scan_gallery_size, self._scan_gallery_K, self._scan_gallery_D
            )

        self._gui_scan_images[index].image = img

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


_ARUCO_IMAGE_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _generate_aruco_image(marker_id: int, *, dictionary: int | None = None, size_px: int = 256) -> np.ndarray:
    """Return an RGB image of the given ArUco marker, cached per (dict, id, size).

    The marker is rendered with a small white quiet-zone border around it (the same
    white margin printed tags have in practice) so the tag edges are clearly
    visible against a colored block in the viewer.
    """
    import cv2

    if dictionary is None:
        dictionary = cv2.aruco.DICT_ARUCO_ORIGINAL

    cache_key = (int(dictionary), int(marker_id), int(size_px))
    cached = _ARUCO_IMAGE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    marker_size = int(size_px * 5 / 6)  # leave ~1/12 white quiet zone on each side
    marker = cv2.aruco.generateImageMarker(aruco_dict, int(marker_id), marker_size)

    padded = np.full((size_px, size_px), 255, dtype=np.uint8)
    pad = (size_px - marker_size) // 2
    padded[pad : pad + marker_size, pad : pad + marker_size] = marker

    rgb = cv2.cvtColor(padded, cv2.COLOR_GRAY2RGB)
    _ARUCO_IMAGE_CACHE[cache_key] = rgb
    return rgb


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

    ``gripper_type="robotiq_2f140"`` downloads (on first use) and processes the
    canonical ROS-Industrial 2F-140 xacro from the
    ``ros-industrial-attic/robotiq`` repo, caches the result under
    ``~/.cache/me235b/``, and loads it with ``yourdfpy``. Any other name is
    forwarded to ``robot_descriptions.loaders.yourdfpy.load_robot_description``
    as ``f"{gripper_type}_description"``.
    """
    if gripper_type == "robotiq_2f140":
        return _load_rosi_robotiq_140(load_meshes=load_meshes)

    with _utf8_open():
        return loader(
            f"{gripper_type}_description",
            load_meshes=load_meshes,
            build_scene_graph=load_meshes,
            load_collision_meshes=False,
            build_collision_scene_graph=False,
        )
