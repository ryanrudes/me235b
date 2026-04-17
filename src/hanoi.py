"""Tower-of-Hanoi orchestration (Lab 3).

Pieces:

- :class:`BoxTag` / :class:`PadTag` — ArUco IDs for boxes and pads.
- :class:`HanoiDetector` — extends :class:`me235b.detector.ArucoDetector` with
  4x4 pose extraction and lab-specified grasp / place geometry.
- :class:`HanoiTask` — owns the :class:`RobotController`, detector, camera, and
  the Hanoi game sequence. Provides ``scan``, ``home``, ``pick_up``,
  ``place_down``, and ``run``.
- :func:`run_hanoi` — Typer-friendly CLI entry point.

Key facts from the Lab 3 spec that this module is built to respect:

1. The camera is rigidly attached to **frame 5** (the frame prior to the
   flange) via the transform ``T5c``. Converting detections to base frame
   therefore uses ``fk_to_frame(q, 5) @ T5c``, not ``fk(q)``.
2. The center of the gripper is **20 cm along frame-6 z** from the last DH
   frame. This is the default ``tool_z_offset`` passed into the kinematics.
3. Towers are base-aligned 20 cm squares with their ArUco tags in the
   bottom-left corner. Grasp / place poses therefore use a known
   base-aligned "down" orientation rather than the (noisy) ArUco rotation.
4. The tag-center-to-block-center offset is ``(block_width/2, tag_width/2,
   -block_thickness/2)`` in the ArUco frame: the block extends **downward**
   from the tag (the tag sits on the top face and its z-axis points out of
   the face, i.e. up).
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
from .transforms import make_T, make_T_rpy


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

# Frame-5 to camera transform per Lab 3 spec (identity rotation, 10.16 cm in -y
# and 8.48 cm in +z, all expressed in frame-5 coordinates).
T5C_DEFAULT: np.ndarray = make_T_rpy([0.0, -0.1016, 0.0848], [0.0, 0.0, 0.0])

# Clean base-aligned top-down grasp orientation (Rx(pi)):
#   gripper +x -> base +x   (jaws close along base x; matches the PDF's
#                            "grab them along the x-axis" recommendation)
#   gripper +y -> base -y
#   gripper +z -> base -z   (approach from above)
GRASP_ORIENTATION: np.ndarray = np.array(
    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
    dtype=float,
)


class HanoiDetector(ArucoDetector):
    """ArUco detector extended with 4x4 tag poses and Lab 3 grasp / place geometry."""

    block_thickness: float = 0.05  # per PDF Part 2 prep (a)
    pad_size: float = 0.20  # per PDF Part 3 prep (b)

    def find_tag_poses(self, frame: np.ndarray) -> list[tuple[int, np.ndarray]]:
        """Return ``[(tag_id, T_cam_marker), ...]`` for every ArUco tag in ``frame``.

        Note: this returns the *camera-frame* pose of each marker, not the base
        frame. Callers that need base-frame poses should multiply by the current
        camera-in-base transform (``T_base_cam @ T_cam_marker``).
        """
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
        """Base -> gripper-center transform for grasping the block at ``T_base_aruco``.

        Returns ``Tbg`` as defined in Lab 3 Part 2 prep (a): "transform between
        the base of the robot and the center of the gripper in the open position
        such that it can be closed to grab the object."

        The block's ArUco tag sits in the bottom-left corner of the top face; the
        prism extends into the block's ``+x`` / ``+y`` / ``-z`` (below the tag).
        The orientation is fixed to :data:`GRASP_ORIENTATION` — we ignore the
        noisy measured ArUco rotation because the PDF guarantees blocks are
        base-frame aligned.
        """
        if tag not in BOX_WIDTHS:
            raise ValueError(f"grasp_pose_for: unknown box tag {tag!r}.")
        block_width = BOX_WIDTHS[tag]

        T_aruco_block = np.eye(4, dtype=float)
        T_aruco_block[0, 3] = block_width / 2
        T_aruco_block[1, 3] = self.marker_length / 2
        T_aruco_block[2, 3] = -self.block_thickness / 2  # block is BELOW the tag face

        T_base_block = np.asarray(T_base_aruco, dtype=float) @ T_aruco_block

        T_grasp = np.eye(4, dtype=float)
        T_grasp[:3, :3] = GRASP_ORIENTATION
        T_grasp[:3, 3] = T_base_block[:3, 3]
        return T_grasp

    def place_pose_for(self, _tag: PadTag, T_base_aruco: np.ndarray) -> np.ndarray:
        """Base -> gripper-center transform for dropping a block onto the given pad.

        The tower is a 20 cm square with its ArUco tag in the bottom-left corner,
        lying flat at ``z ~ 0`` with base-frame-aligned orientation. We take only
        the tag's position, offset in the **base** frame by ``(pad_size/2,
        pad_size/2, block_thickness/2)``, and stamp on a clean downward grasp
        orientation so the held block ends up resting on the pad.
        """
        aruco_pos = np.asarray(T_base_aruco, dtype=float)[:3, 3]
        pad_center_xy = aruco_pos + np.array([self.pad_size / 2, self.pad_size / 2, 0.0], dtype=float)
        # Place so the block rests flat on the pad (gripper center = block center).
        place_pos = pad_center_xy + np.array([0.0, 0.0, self.block_thickness / 2], dtype=float)

        T_place = np.eye(4, dtype=float)
        T_place[:3, :3] = GRASP_ORIENTATION
        T_place[:3, 3] = place_pos
        return T_place

    def grasp_dict(self, detections: dict[BoxTag, np.ndarray]) -> dict[BoxTag, np.ndarray]:
        return {tag: self.grasp_pose_for(tag, T) for tag, T in detections.items()}

    def place_dict(self, detections: dict[PadTag, np.ndarray]) -> dict[PadTag, np.ndarray]:
        return {tag: self.place_pose_for(tag, T) for tag, T in detections.items()}


def _default_home_pose() -> np.ndarray:
    """Home pose: 10 cm above the base's y = -0.5 region, gripper pointing down."""
    T = np.eye(4, dtype=float)
    T[:3, :3] = GRASP_ORIENTATION
    T[:3, 3] = [0.30, -0.40, 0.40]
    return T


def _default_scan_points(scan_height: float = 0.4) -> np.ndarray:
    """Six scan positions tiled across the tag workspace ``x in [-0.7, 0.7], y in [-1, 0]``."""
    return np.array(
        [
            [-0.35, -0.33, scan_height],
            [0.0, -0.33, scan_height],
            [0.35, -0.33, scan_height],
            [-0.35, -0.67, scan_height],
            [0.0, -0.67, scan_height],
            [0.35, -0.67, scan_height],
        ],
        dtype=float,
    )


class HanoiTask:
    """Tower-of-Hanoi orchestrator.

    Internal model: :attr:`pad_stacks` maps each pad to a list of blocks on it,
    ordered bottom -> top. Grasp / place poses are computed dynamically from
    this model (plus the pad's scanned base-frame position) so there's no such
    thing as a "stale" grasp pose — after every successful move we update the
    stacks, and the next move reaches for the block's *new* location.

    Typical workflow::

        task = HanoiTask(controller, detector, camera=cam)
        task.home()
        task.scan()                 # populates pad_centers and pad_stacks
        task.solve_tower_of_hanoi() # runs the full 7-move solution

    :meth:`run` bundles ``home -> scan -> solve_tower_of_hanoi -> home`` together.
    """

    def __init__(
        self,
        controller: RobotController,
        detector: HanoiDetector,
        *,
        camera: Any | None = None,
        T_home: np.ndarray | None = None,
        T5c: np.ndarray | None = None,
        scan_points: np.ndarray | None = None,
        approach_height: float = 0.08,
    ) -> None:
        self.controller = controller
        self.detector = detector
        self.camera = camera
        self.T_home = _default_home_pose() if T_home is None else np.asarray(T_home, dtype=float)
        self.T5c = T5C_DEFAULT if T5c is None else np.asarray(T5c, dtype=float)
        self.scan_points = _default_scan_points() if scan_points is None else np.asarray(scan_points, dtype=float)
        self.approach_height = float(approach_height)

        # Camera-side kinematics: IK targets are camera poses ~= T_b6 @ T5c.
        # (The exact camera pose is T_B5 @ T5c, which differs by the joint-6
        # transform; we accept this approximation when *moving* to a scan point
        # and compute the *actual* camera pose from frame-5 FK after the move.)
        self.camera_kin = UR10e(
            T6t=self.T5c,
            joint_limits_rad=controller.kin.joint_limits_rad,
            joint_limit_margin_rad=controller.kin.joint_limit_margin_rad,
            external_safety_filter=controller.kin.external_safety_filter,
        )

        self._box_handles: dict[BoxTag, str] = {}
        self._pad_handles: dict[PadTag, str] = {}
        self._holding: BoxTag | None = None

        # Internal world model.
        self.pad_centers: dict[PadTag, np.ndarray] = {}
        self.pad_stacks: dict[PadTag, list[BoxTag]] = {p: [] for p in PadTag}

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

    def _current_T_base_cam(self) -> np.ndarray:
        """Camera-in-base transform computed from the controller's live joint state."""
        q = self.controller.current_q_class
        T_b5 = self.controller.kin.fk_to_frame(q, 5)
        return T_b5 @ self.T5c

    def scan(self) -> tuple[dict[BoxTag, np.ndarray], dict[PadTag, np.ndarray]]:
        """Drive the camera through each scan point, collect base-frame tag poses,
        then update the internal world model (pad centers + stacks)."""
        box_detections: dict[BoxTag, np.ndarray] = {}
        pad_detections: dict[PadTag, np.ndarray] = {}

        for scan_point in self.scan_points:
            T_bc = make_T_rpy(scan_point, [np.pi, 0.0, 0.0])
            ok, _theta, info = self.controller.move_to_pose(T_bc, kinematics=self.camera_kin)
            if not ok:
                if self.controller.verbose:
                    print(f"[hanoi] scan move IK failed at {scan_point}: {info.get('message','')}")
                continue

            if self.camera is None:
                continue

            frame = self._grab_frame()
            if frame is None:
                continue

            T_base_cam = self._current_T_base_cam()

            for tag_id, T_cam_marker in self.detector.find_tag_poses(frame):
                T_base_marker = T_base_cam @ T_cam_marker

                try:
                    box = BoxTag(tag_id)
                    box_detections[box] = T_base_marker
                    continue
                except ValueError:
                    pass
                try:
                    pad = PadTag(tag_id)
                    pad_detections[pad] = T_base_marker
                    continue
                except ValueError:
                    pass
                if self.controller.verbose:
                    print(f"[hanoi] ignoring non-Hanoi tag id={tag_id}")

        self._ingest_detections(box_detections, pad_detections)
        return box_detections, pad_detections

    def _ingest_detections(
        self,
        box_detections: dict[BoxTag, np.ndarray],
        pad_detections: dict[PadTag, np.ndarray],
    ) -> None:
        """Populate :attr:`pad_centers` and :attr:`pad_stacks` from a scan.

        Pad center = ArUco corner + (pad_size/2, pad_size/2, 0) in base frame.
        Each block is assigned to whichever pad center it's closest to (xy), then
        each pad's blocks are ordered largest-at-bottom (Hanoi invariant).
        """
        self.pad_centers.clear()
        for tag, T_aruco in pad_detections.items():
            p = np.asarray(T_aruco, dtype=float)[:3, 3]
            self.pad_centers[tag] = p + np.array(
                [self.detector.pad_size / 2, self.detector.pad_size / 2, 0.0],
                dtype=float,
            )

        self.pad_stacks = {p: [] for p in PadTag}
        if not self.pad_centers:
            return

        for box, T_aruco in box_detections.items():
            box_pos = np.asarray(T_aruco, dtype=float)[:3, 3]
            closest_pad = min(
                self.pad_centers,
                key=lambda p: float(np.linalg.norm(self.pad_centers[p][:2] - box_pos[:2])),
            )
            self.pad_stacks[closest_pad].append(box)

        # Largest block on the bottom (Hanoi invariant).
        for stack in self.pad_stacks.values():
            stack.sort(key=lambda b: -BOX_WIDTHS[b])

    def _pose_on_stack(self, pad: PadTag, stack_index: int) -> np.ndarray:
        """Gripper-center pose for grabbing / placing at position ``stack_index`` on ``pad``.

        ``stack_index`` is the 0-based index of the block from the bottom. The gripper
        center sits at the block's center, so z = (stack_index + 0.5) * block_thickness.
        """
        if pad not in self.pad_centers:
            raise RuntimeError(f"_pose_on_stack: pad {pad.name} has no known center (run scan first).")

        bt = self.detector.block_thickness
        center_xy = self.pad_centers[pad]
        z = stack_index * bt + bt / 2

        T = np.eye(4, dtype=float)
        T[:3, :3] = GRASP_ORIENTATION
        T[:3, 3] = [float(center_xy[0]), float(center_xy[1]), float(z)]
        return T

    def grasp_pose_for_top(self, pad: PadTag) -> tuple[BoxTag, np.ndarray]:
        """Return ``(block_on_top, gripper_pose)`` for the current top of ``pad``."""
        stack = self.pad_stacks.get(pad, [])
        if not stack:
            raise RuntimeError(f"grasp_pose_for_top: pad {pad.name} is empty.")
        return stack[-1], self._pose_on_stack(pad, len(stack) - 1)

    def place_pose_for_top(self, pad: PadTag) -> np.ndarray:
        """Return the gripper pose for landing the next block on ``pad``."""
        return self._pose_on_stack(pad, len(self.pad_stacks.get(pad, [])))

    def _hover_pose(self, target: np.ndarray) -> np.ndarray:
        """Return ``target`` shifted up by :attr:`approach_height` in the base z axis."""
        out = np.asarray(target, dtype=float).copy()
        out[2, 3] += self.approach_height
        return out

    def pick_up(self, tag: BoxTag, grasp_pose: np.ndarray) -> bool:
        """Hover -> descend -> close -> lift sequence for ``tag`` at ``grasp_pose``.

        Returns ``True`` if every motion step's IK succeeded.
        """
        hover = self._hover_pose(grasp_pose)

        for label, T in (("hover-above", hover), ("descend", grasp_pose)):
            ok, _, info = self.controller.move_to_pose(T)
            if not ok:
                if self.controller.verbose:
                    print(f"[hanoi] pick_up {tag.name} {label} IK failed: {info.get('message','')}")
                return False

        self.controller.gripper_close(grasped_width_m=BOX_WIDTHS.get(tag))
        if tag in self._box_handles:
            self.renderer.attach_to_end_effector(self._box_handles[tag])
            self._holding = tag

        ok, _, info = self.controller.move_to_pose(hover)
        if not ok and self.controller.verbose:
            print(f"[hanoi] pick_up {tag.name} lift IK failed: {info.get('message','')}")
        return ok

    def place_down(self, tag: PadTag, place_pose: np.ndarray) -> bool:
        """Hover -> descend -> open -> lift sequence, placing the held block on ``tag``."""
        hover = self._hover_pose(place_pose)

        for label, T in (("hover-above", hover), ("descend", place_pose)):
            ok, _, info = self.controller.move_to_pose(T)
            if not ok:
                if self.controller.verbose:
                    print(f"[hanoi] place_down {tag.name} {label} IK failed: {info.get('message','')}")
                return False

        if self._holding is not None and self._holding in self._box_handles:
            self.renderer.detach_from_end_effector(self._box_handles[self._holding])
        self._holding = None
        self.controller.gripper_open()

        ok, _, info = self.controller.move_to_pose(hover)
        if not ok and self.controller.verbose:
            print(f"[hanoi] place_down {tag.name} lift IK failed: {info.get('message','')}")
        return ok

    def populate_scene(self) -> None:
        """Render every known pad and every block (at its *current* stacked pose).

        Uses the internal :attr:`pad_stacks` model, so whatever state the scan
        produced is what the viewer shows. Stacks are drawn centered on the pad.
        """
        self._box_handles.clear()
        self._pad_handles.clear()

        bt = self.detector.block_thickness

        for pad, center_xy in self.pad_centers.items():
            T_visual = np.eye(4, dtype=float)
            T_visual[:3, 3] = [float(center_xy[0]), float(center_xy[1]), -0.002]
            handle = self.renderer.add_box(
                T_visual,
                dimensions=(self.detector.pad_size, self.detector.pad_size, 0.004),
                label=f"pad_{pad.name.lower()}",
                color=PAD_COLORS[pad],
            )
            self._pad_handles[pad] = handle

        for pad, stack in self.pad_stacks.items():
            if pad not in self.pad_centers:
                continue
            center_xy = self.pad_centers[pad]
            for i, box in enumerate(stack):
                width = BOX_WIDTHS[box]
                T_visual = np.eye(4, dtype=float)
                T_visual[:3, 3] = [float(center_xy[0]), float(center_xy[1]), i * bt + bt / 2]
                handle = self.renderer.add_box(
                    T_visual,
                    dimensions=(width, width, bt),
                    label=f"box_{box.name.lower()}",
                    color=BOX_COLORS[box],
                )
                self._box_handles[box] = handle

    def _fake_detections(self) -> tuple[dict[BoxTag, np.ndarray], dict[PadTag, np.ndarray]]:
        """Synthetic detections matching a standard Tower-of-Hanoi starting state.

        All three blocks start stacked on ``START_PAD``, middle and end pads empty.
        """
        bt = self.detector.block_thickness
        tw = self.detector.marker_length
        R_id = np.eye(3, dtype=float)

        # Pad ArUco corners (bottom-left corner in each pad's 20 cm square).
        pad_offset = np.array([self.detector.pad_size / 2, self.detector.pad_size / 2, 0.0], dtype=float)
        pad_centers = {
            PadTag.START_PAD: np.array([-0.20, -0.85, 0.0], dtype=float),
            PadTag.MIDDLE_PAD: np.array([0.10, -0.85, 0.0], dtype=float),
            PadTag.END_PAD: np.array([0.30, -0.85, 0.0], dtype=float),
        }
        pads = {tag: make_T(center - pad_offset, R_id) for tag, center in pad_centers.items()}

        # Stack all three blocks on START_PAD, largest on bottom:
        # LARGE at index 0, MEDIUM at 1, SMALL at 2.
        start_xy = pad_centers[PadTag.START_PAD][:2]
        stack_order = [BoxTag.LARGE_BOX, BoxTag.MEDIUM_BOX, BoxTag.SMALL_BOX]
        boxes: dict[BoxTag, np.ndarray] = {}
        for i, tag in enumerate(stack_order):
            w = BOX_WIDTHS[tag]
            # Block center (what grasp_pose_for should land at).
            center = np.array([start_xy[0], start_xy[1], i * bt + bt / 2], dtype=float)
            # Back out the ArUco position: grasp_pose_for applies R_tool @ (w/2, tw/2, -bt/2).
            local = np.array([w / 2, tw / 2, -bt / 2], dtype=float)
            aruco_pos = center - R_id @ local
            boxes[tag] = make_T(aruco_pos, R_id)
        return boxes, pads

    def move_top(self, src: PadTag, dst: PadTag) -> bool:
        """Move the top block of ``src`` to the top of ``dst``.

        Grasp and place poses are computed from the internal world model, so
        this always reaches for the block that's *actually* on top right now.
        """
        stack = self.pad_stacks.get(src, [])
        if not stack:
            if self.controller.verbose:
                print(f"[hanoi] move_top: {src.name} is empty, skipping.")
            return False

        top_box, grasp_pose = self.grasp_pose_for_top(src)
        place_pose = self.place_pose_for_top(dst)

        if self.controller.verbose:
            print(f"[hanoi] {top_box.name}: {src.name} -> {dst.name}")

        if not self.pick_up(top_box, grasp_pose):
            return False
        self.home()
        if not self.place_down(dst, place_pose):
            return False
        self.home()

        # Update the world model.
        self.pad_stacks[src].pop()
        self.pad_stacks[dst].append(top_box)
        return True

    def solve_tower_of_hanoi(
        self,
        *,
        source: PadTag = PadTag.START_PAD,
        target: PadTag = PadTag.END_PAD,
        auxiliary: PadTag = PadTag.MIDDLE_PAD,
    ) -> bool:
        """Solve Tower of Hanoi: move every block from ``source`` to ``target``.

        Uses the classical recursive algorithm via :meth:`move_top`. Any number
        of blocks on ``source`` is supported (the method reads the current stack
        depth from :attr:`pad_stacks`).
        """
        n = len(self.pad_stacks.get(source, []))
        if n == 0:
            if self.controller.verbose:
                print(f"[hanoi] solve_tower_of_hanoi: {source.name} is empty.")
            return True

        success = True

        def recurse(count: int, src: PadTag, dst: PadTag, aux: PadTag) -> None:
            nonlocal success
            if count == 0 or not success:
                return
            recurse(count - 1, src, aux, dst)
            if not success:
                return
            if not self.move_top(src, dst):
                success = False
                return
            recurse(count - 1, aux, dst, src)

        recurse(n, source, target, auxiliary)
        return success

    def run(self) -> None:
        """``home -> scan -> populate_scene -> solve_tower_of_hanoi -> home``."""
        self.home()

        if self.camera is None and self.controller.simulate:
            if self.controller.verbose:
                print("[hanoi] no camera + simulate=True -> using synthetic detections.")
            box_detections, pad_detections = self._fake_detections()
            self._ingest_detections(box_detections, pad_detections)
        else:
            self.scan()

        missing_pads = [p for p in PadTag if p not in self.pad_centers]
        if missing_pads and not self.controller.simulate:
            raise ValueError(f"HanoiTask.run: missing required pads: {missing_pads}")

        self.populate_scene()
        self.solve_tower_of_hanoi()
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
    tool_z_offset: Annotated[float, typer.Option(help="Distance from frame-6 to gripper center (meters). The Lab 3 spec says this is 0.20.")] = 0.20,
    approach_height: Annotated[float, typer.Option(help="Hover height above each grasp / place pose (meters).")] = 0.08,
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

        task = HanoiTask(controller, detector, camera=camera, approach_height=approach_height)
        task.run()
    finally:
        if camera is not None:
            camera.release()
        controller.close()
        if renderer is not None:
            renderer.close(wait=(render is RenderMode.viser))
