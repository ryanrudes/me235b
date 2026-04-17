"""Tower-of-Hanoi orchestration (Lab 3).

Pieces:

- :class:`BoxTag` / :class:`PadTag` — ArUco IDs for boxes and pads.
- :class:`HanoiDetector` — extends :class:`me235b.detector.ArucoDetector`
  with 4x4 pose extraction + grasp / place geometry.
- :class:`HanoiTask` — owns the :class:`RobotController`, detector, camera,
  and the plan for a short Hanoi game. Provides ``scan``, ``home``,
  ``move_to_waypoint``, ``pick_up``, ``place_down``, and ``run``.
- :func:`run_hanoi` — Typer-friendly CLI entry point.
"""

from __future__ import annotations

from enum import Enum, IntEnum
from typing import Any

import numpy as np
import typer
from typing_extensions import Annotated

from .detector import ArucoDetector
from .kinematics import UR10e
from .robot import RobotController
from .sim import NullRenderer, SimulationRenderer
from .transforms import make_T, make_T_rpy, tool_orientation_tilted


class BoxTag(IntEnum):
    SMALL_BOX = 3
    MEDIUM_BOX = 4
    LARGE_BOX = 5


class PadTag(IntEnum):
    START_PAD = 0
    MIDDLE_PAD = 1
    END_PAD = 2


BOX_WIDTHS: dict[BoxTag, float] = {
    BoxTag.SMALL_BOX: 0.06,
    BoxTag.MEDIUM_BOX: 0.10,
    BoxTag.LARGE_BOX: 0.12,
}

BOX_COLORS: dict[BoxTag, tuple[int, int, int]] = {
    BoxTag.SMALL_BOX: (220, 120, 120),
    BoxTag.MEDIUM_BOX: (120, 200, 130),
    BoxTag.LARGE_BOX: (130, 150, 220),
}

PAD_COLORS: dict[PadTag, tuple[int, int, int]] = {
    PadTag.START_PAD: (220, 220, 140),
    PadTag.MIDDLE_PAD: (220, 180, 130),
    PadTag.END_PAD: (170, 220, 180),
}

ALL_HANOI_TAGS: tuple[BoxTag | PadTag, ...] = tuple(BoxTag) + tuple(PadTag)

# Fixed camera offset from frame 5 (z-offset + camera y-translation per Lab 3 rig).
T5C_DEFAULT: np.ndarray = make_T_rpy([0.0, -0.1016, 0.0848], [0.0, 0.0, 0.0])


class HanoiDetector(ArucoDetector):
    """ArUco detector extended with 4x4 tag poses and grasp/place geometry."""

    block_thickness: float = 0.05
    pad_size: float = 0.20

    def find_tag_poses(self, frame: np.ndarray) -> list[tuple[int, np.ndarray]]:
        """Return ``[(tag_id, T_cam_marker), ...]`` for every ArUco tag in ``frame``."""
        detections = self.find_tags(frame)
        results: list[tuple[int, np.ndarray]] = []
        for tag_id, rvec, tvec in detections:
            import cv2

            R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))
            T = np.eye(4, dtype=float)
            T[:3, :3] = R
            T[:3, 3] = np.asarray(tvec, dtype=float).reshape(3)
            results.append((int(tag_id), T))
        return results

    def grasp_pose_for(self, tag: BoxTag, T_base_aruco: np.ndarray) -> np.ndarray:
        """Base-frame pose for the center of the block whose ArUco tag is at ``T_base_aruco``."""
        if tag not in BOX_WIDTHS:
            raise ValueError(f"grasp_pose_for: unknown box tag {tag!r}.")
        block_width = BOX_WIDTHS[tag]

        T_aruco_block = np.eye(4, dtype=float)
        T_aruco_block[0, 3] = block_width / 2
        T_aruco_block[1, 3] = self.marker_length / 2
        T_aruco_block[2, 3] = self.block_thickness / 2
        return np.asarray(T_base_aruco, dtype=float) @ T_aruco_block

    def place_pose_for(self, _tag: PadTag, T_base_aruco: np.ndarray) -> np.ndarray:
        """Base-frame pose for the center of a pad whose corner ArUco tag is at ``T_base_aruco``."""
        offset = self.pad_size / 2
        return np.asarray(T_base_aruco, dtype=float) @ make_T_rpy([offset, offset, 0.0], [0.0, 0.0, 0.0])

    def grasp_dict(self, detections: dict[BoxTag, np.ndarray]) -> dict[BoxTag, np.ndarray]:
        return {tag: self.grasp_pose_for(tag, T) for tag, T in detections.items()}

    def place_dict(self, detections: dict[PadTag, np.ndarray]) -> dict[PadTag, np.ndarray]:
        return {tag: self.place_pose_for(tag, T) for tag, T in detections.items()}


def _default_home_pose() -> np.ndarray:
    """Reachable home pose used by :class:`HanoiTask` when the caller doesn't override it."""
    T = np.eye(4, dtype=float)
    T[:3, :3] = tool_orientation_tilted()
    T[:3, 3] = [0.40, -0.10, 0.10]
    return T


def _default_scan_points(scan_height: float = 0.4) -> np.ndarray:
    """Six scan positions in the base frame, tiled to cover the box/pad workspace."""
    return np.array(
        [
            [-0.7 / 2, -1 / 3, scan_height],
            [0.0, -1 / 3, scan_height],
            [0.7 / 2, -1 / 3, scan_height],
            [-0.7 / 2, 2 / 3, scan_height],
            [0.0, 2 / 3, scan_height],
            [0.7 / 2, 2 / 3, scan_height],
        ],
        dtype=float,
    )


class HanoiTask:
    """Tower-of-Hanoi orchestrator around a :class:`RobotController` + :class:`HanoiDetector`."""

    def __init__(
        self,
        controller: RobotController,
        detector: HanoiDetector,
        *,
        camera: Any | None = None,
        T_home: np.ndarray | None = None,
        T5c: np.ndarray | None = None,
        scan_points: np.ndarray | None = None,
        lift_distance: float = 0.05,
    ) -> None:
        self.controller = controller
        self.detector = detector
        self.camera = camera
        self.T_home = _default_home_pose() if T_home is None else np.asarray(T_home, dtype=float)
        self.T5c = T5C_DEFAULT if T5c is None else np.asarray(T5c, dtype=float)
        self.scan_points = _default_scan_points() if scan_points is None else np.asarray(scan_points, dtype=float)
        self.lift_distance = float(lift_distance)

        # Dedicated kinematics for camera positioning (same robot, different tool offset).
        self.camera_kin = UR10e(
            T6t=self.T5c,
            joint_limits_rad=controller.kin.joint_limits_rad,
            joint_limit_margin_rad=controller.kin.joint_limit_margin_rad,
            external_safety_filter=controller.kin.external_safety_filter,
        )

        self._box_handles: dict[BoxTag, str] = {}
        self._pad_handles: dict[PadTag, str] = {}
        self._holding: BoxTag | None = None

    @property
    def renderer(self) -> SimulationRenderer:
        return self.controller.renderer

    def home(self) -> dict:
        """Drive the gripper to the configured home pose."""
        return self.controller.home(self.T_home, verify_fk=True)

    def _grab_frame(self) -> np.ndarray | None:
        if self.camera is None:
            return None
        ok, frame = self.camera.read()
        if not ok:
            raise RuntimeError("HanoiTask.scan: camera.read() failed.")
        return frame

    def scan(self) -> tuple[dict[BoxTag, np.ndarray], dict[PadTag, np.ndarray]]:
        """Drive the camera through each scan point and collect tag detections."""
        box_detections: dict[BoxTag, np.ndarray] = {}
        pad_detections: dict[PadTag, np.ndarray] = {}

        for scan_point in self.scan_points:
            T_bc = make_T_rpy(scan_point, [np.pi, 0.0, 0.0])
            ok, _theta, info = self.controller.move_to_pose(T_bc, kinematics=self.camera_kin)
            if not ok and self.controller.verbose:
                print(f"[hanoi] scan move IK failed: {info.get('message','')}")
                continue

            if self.camera is None:
                continue

            frame = self._grab_frame()
            if frame is None:
                continue

            for tag_id, T_cam_marker in self.detector.find_tag_poses(frame):
                try:
                    box = BoxTag(tag_id)
                    box_detections[box] = T_cam_marker
                    continue
                except ValueError:
                    pass
                try:
                    pad = PadTag(tag_id)
                    pad_detections[pad] = T_cam_marker
                    continue
                except ValueError:
                    pass
                if self.controller.verbose:
                    print(f"[hanoi] ignoring non-Hanoi tag id={tag_id}")

        return box_detections, pad_detections

    def move_to_waypoint(
        self,
        waypoints: dict[BoxTag | PadTag, np.ndarray],
        tag: BoxTag | PadTag,
        *,
        z_offset: float = 0.0,
    ) -> tuple[bool, np.ndarray, dict]:
        if tag not in waypoints:
            raise KeyError(f"move_to_waypoint: tag {tag!r} not in waypoints.")
        T = waypoints[tag] @ make_T_rpy([0.0, 0.0, z_offset], [0.0, 0.0, 0.0])
        return self.controller.move_to_pose(T)

    def pick_up(self, tag: BoxTag | None = None, *, lift_distance: float | None = None) -> None:
        self.controller.gripper_close()
        if tag is not None and tag in self._box_handles:
            self.renderer.attach_to_end_effector(self._box_handles[tag])
            self._holding = tag
        self.controller.translate(0.0, 0.0, self.lift_distance if lift_distance is None else lift_distance)

    def place_down(self, *, lift_distance: float | None = None) -> None:
        if self._holding is not None and self._holding in self._box_handles:
            self.renderer.detach_from_end_effector(self._box_handles[self._holding])
        self._holding = None
        self.controller.gripper_open()
        self.controller.translate(0.0, 0.0, self.lift_distance if lift_distance is None else lift_distance)

    def populate_scene(
        self,
        box_detections: dict[BoxTag, np.ndarray],
        pad_detections: dict[PadTag, np.ndarray],
    ) -> None:
        """Add boxes and pads to the attached renderer so they show up in the viewer.

        The grasp / place poses carry the ArUco detection's orientation (which, in the
        lab's convention, mirrors the tilted tool approach). For the visualization we
        override that orientation so blocks and pads sit flat in the world frame,
        regardless of whatever funky angle the tag was detected at.
        """
        self._box_handles.clear()
        self._pad_handles.clear()

        for tag, T_aruco in box_detections.items():
            width = BOX_WIDTHS[tag]
            T_block = self.detector.grasp_pose_for(tag, T_aruco)
            T_visual = np.eye(4, dtype=float)
            T_visual[:3, 3] = T_block[:3, 3]
            handle = self.renderer.add_box(
                T_visual,
                dimensions=(width, width, self.detector.block_thickness),
                label=f"box_{tag.name.lower()}",
                color=BOX_COLORS[tag],
            )
            self._box_handles[tag] = handle

        for tag, T_aruco in pad_detections.items():
            T_pad = self.detector.place_pose_for(tag, T_aruco)
            T_visual = np.eye(4, dtype=float)
            T_visual[:3, 3] = T_pad[:3, 3]
            T_visual[2, 3] -= 0.002  # just below the pen touch plane so it doesn't z-fight
            handle = self.renderer.add_box(
                T_visual,
                dimensions=(self.detector.pad_size, self.detector.pad_size, 0.004),
                label=f"pad_{tag.name.lower()}",
                color=PAD_COLORS[tag],
            )
            self._pad_handles[tag] = handle

    def _fake_detections(self) -> tuple[dict[BoxTag, np.ndarray], dict[PadTag, np.ndarray]]:
        """Synthetic detections used for simulate/no-camera dry runs.

        ArUco tag orientations are set to the tilted tool orientation so the grasp /
        place poses inherit a top-down-ish approach (matching the lab's convention
        that the tag's Z-axis points toward the tool). The rendered block / pad
        orientations are overridden to identity in :meth:`populate_scene` so the
        objects still appear flat in the viewer.

        Positions are chosen so:

        - the three boxes are at least ``largest_box_width + 3 cm`` apart in Y so the
          visualization doesn't show them overlapping, and
        - every grasp pose (block center) and place pose (pad center) has a clean IK
          solution from the default home seed.
        """
        R_tool = tool_orientation_tilted()
        block_thickness = self.detector.block_thickness

        # Desired block-center positions (what the viewer will render).
        block_centers = {
            BoxTag.SMALL_BOX: np.array([0.40, -0.20, block_thickness / 2], dtype=float),
            BoxTag.MEDIUM_BOX: np.array([0.40, -0.05, block_thickness / 2], dtype=float),
            BoxTag.LARGE_BOX: np.array([0.40, 0.12, block_thickness / 2], dtype=float),
        }
        boxes: dict[BoxTag, np.ndarray] = {}
        for tag, center in block_centers.items():
            width = BOX_WIDTHS[tag]
            # grasp_pose_for(tag, T) = T @ T_aruco_block, where T_aruco_block has
            # translation (w/2, marker/2, thickness/2) along the aruco axes. To have the
            # grasp pose land at ``center`` we solve for the aruco position. Because
            # T_aruco has rotation R_tool, the offset must be R_tool @ local_offset.
            local_offset = np.array([width / 2, self.detector.marker_length / 2, block_thickness / 2], dtype=float)
            aruco_pos = center - R_tool @ local_offset
            boxes[tag] = make_T(aruco_pos, R_tool)

        # Desired pad-center positions. Pads sit outboard of the boxes but close enough
        # that the tilted-tool IK (with the Lab 3 gripper's ~15cm Z offset) has an
        # analytic solution at every pad.
        pad_centers = {
            PadTag.START_PAD: np.array([0.50, -0.25, 0.0], dtype=float),
            PadTag.MIDDLE_PAD: np.array([0.50, -0.08, 0.0], dtype=float),
            PadTag.END_PAD: np.array([0.50, 0.16, 0.0], dtype=float),
        }
        pad_local = np.array([self.detector.pad_size / 2, self.detector.pad_size / 2, 0.0], dtype=float)
        pads = {tag: make_T(center - R_tool @ pad_local, R_tool) for tag, center in pad_centers.items()}
        return boxes, pads

    def run(self) -> None:
        """A short Hanoi-style sequence: small -> middle pad, then medium -> end pad."""
        self.home()

        if self.camera is None and self.controller.simulate:
            if self.controller.verbose:
                print("[hanoi] no camera + simulate=True -> using synthetic detections.")
            box_detections, pad_detections = self._fake_detections()
        else:
            box_detections, pad_detections = self.scan()

        missing = [t for t in ALL_HANOI_TAGS if t not in box_detections and t not in pad_detections]
        if missing and not self.controller.simulate:
            raise ValueError(f"HanoiTask.run: missing required tags: {missing}")

        box_waypoints = self.detector.grasp_dict(box_detections)
        pad_waypoints = self.detector.place_dict(pad_detections)

        self.populate_scene(box_detections, pad_detections)

        def safe_move(waypoints, tag):
            if tag not in waypoints:
                if self.controller.verbose:
                    print(f"[hanoi] skipping {tag!r} (not detected)")
                return False
            ok, _, _ = self.move_to_waypoint(waypoints, tag)
            return ok

        if safe_move(box_waypoints, BoxTag.SMALL_BOX):
            self.pick_up(BoxTag.SMALL_BOX)
            self.home()
            if safe_move(pad_waypoints, PadTag.MIDDLE_PAD):
                self.place_down()
                self.home()

        if safe_move(box_waypoints, BoxTag.MEDIUM_BOX):
            self.pick_up(BoxTag.MEDIUM_BOX)
            self.home()
            if safe_move(pad_waypoints, PadTag.END_PAD):
                self.place_down()
                self.home()


class RenderMode(str, Enum):
    """Rendering backend for ``run_hanoi``."""

    none = "none"
    viser = "viser"


def _build_renderer(mode: RenderMode, kin: UR10e) -> SimulationRenderer | None:
    if mode is RenderMode.none:
        return None
    if mode is RenderMode.viser:
        from .sim import ViserRenderer

        return ViserRenderer(kin)
    raise ValueError(f"Unknown render mode: {mode!r}")


def run_hanoi(
    ip: Annotated[str, typer.Option(help="IP address of the UR10e controller.")] = "192.168.0.2",
    simulate: Annotated[bool, typer.Option("--simulate/--no-simulate", help="Skip robot + camera I/O.")] = True,
    dry_run: Annotated[bool, typer.Option("--dry-run/--no-dry-run", help="Connect to robot but print commands instead of executing.")] = False,
    camera_index: Annotated[int, typer.Option(help="Index passed to cv2.VideoCapture.")] = 0,
    render: Annotated[RenderMode, typer.Option(help="Rendering backend: 'none' or 'viser'.")] = RenderMode.none,
    step_duration: Annotated[float, typer.Option(help="Seconds to animate each joint target in the renderer.")] = 0.4,
    tool_z_offset: Annotated[float, typer.Option(help="Gripper length along frame-6 z (meters). 0 targets the flange; 0.15 is a typical 2F gripper.")] = 0.15,
) -> None:
    """Run the Lab 3 Tower-of-Hanoi sequence."""
    kin = UR10e(T6t=make_T([0.0, 0.0, float(tool_z_offset)]))
    renderer = _build_renderer(render, kin)
    controller = RobotController(
        kin,
        simulate=simulate,
        dry_run=dry_run,
        renderer=renderer,
        step_duration=step_duration,
    )
    detector = HanoiDetector()

    camera = None
    try:
        if not simulate:
            controller.connect(ip)
            import cv2

            camera = cv2.VideoCapture(camera_index)
            if not camera.isOpened():
                raise RuntimeError(f"run_hanoi: cv2.VideoCapture({camera_index}) failed to open.")

        task = HanoiTask(controller, detector, camera=camera)
        task.run()
    finally:
        if camera is not None:
            camera.release()
        controller.close()
        if renderer is not None:
            renderer.close(wait=(render is RenderMode.viser))
