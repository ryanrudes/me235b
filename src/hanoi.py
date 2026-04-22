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

from collections import defaultdict
from enum import Enum, IntEnum
import threading
import time
from typing import Any, Sequence

import numpy as np
import typer
from typing_extensions import Annotated

from .detector import ArucoDetector
from .kinematics import UR10e
from .robot import RobotController
from .sim import (
    DemoAbort,
    LAB3_CAMERA_IMAGE_H,
    LAB3_CAMERA_IMAGE_W,
    NullRenderer,
    SimulationRenderer,
    ViserRenderer,
)
from .transforms import fuse_rigid_transforms, make_T, make_T_rpy


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

# Thin pad slab in :meth:`HanoiTask.populate_scene` (must match :meth:`pad_center_from_marker_pose`).
PAD_SLAB_THICKNESS_M: float = 0.004
PAD_SLAB_CENTER_Z_M: float = -0.002
# Local tag offsets in :meth:`HanoiTask.populate_scene` (pad / block parent frames).
PAD_TAG_LOCAL_Z_M: float = 0.003
BLOCK_TAG_Z_EPSILON_M: float = 0.001

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

DEFAULT_SCAN_HEIGHT = 0.6

def grasp_orientation_from_measured_tag(
    T_base_aruco: np.ndarray,
    *,
    fallback: np.ndarray = GRASP_ORIENTATION,
) -> np.ndarray:
    """Gripper rotation (columns = gripper x,y,z expressed in base) from a tag pose.

    Uses the ArUco marker frame from ``T_base_aruco`` (OpenCV convention: tag +z
    is normal to the marker plane, pointing out of the pattern). Assumes the tag
    lies on the block's top face so that **gripper +z** aligns with **−tag +z**
    (top-down approach) and **gripper +x** follows the tag's in-plane x axis
    (jaw closing direction per the lab PDF). Falls back to ``fallback`` if the
    tag x axis is nearly parallel to the approach direction (degenerate).
    """
    Rm = np.asarray(T_base_aruco, dtype=float)[:3, :3]
    e3 = np.array([0.0, 0.0, 1.0], dtype=float)
    n_up = Rm @ e3
    nlen = float(np.linalg.norm(n_up))
    if nlen < 1e-9:
        return np.asarray(fallback, dtype=float).copy()
    n_up = n_up / nlen
    ez = -n_up
    tx = Rm @ np.array([1.0, 0.0, 0.0], dtype=float)
    ex = tx - float(np.dot(tx, ez)) * ez
    exn = float(np.linalg.norm(ex))
    if exn < 1e-3:
        return np.asarray(fallback, dtype=float).copy()
    ex = ex / exn
    ey = np.cross(ez, ex)
    eyn = float(np.linalg.norm(ey))
    if eyn < 1e-9:
        return np.asarray(fallback, dtype=float).copy()
    ey = ey / eyn
    return np.stack([ex, ey, ez], axis=1)


class HanoiDetector(ArucoDetector):
    """ArUco detector extended with 4x4 tag poses and Lab 3 grasp / place geometry."""

    block_thickness: float = 0.05  # per PDF Part 2 prep (a)
    pad_size: float = 0.20  # per PDF Part 3 prep (b)

    def pad_center_from_marker_pose(self, T_base_marker: np.ndarray) -> np.ndarray:
        """Pad geometric center in base frame from a pad marker :math:`4\\times4` pose.

        ``solvePnP`` / :meth:`find_tag_poses` put the marker origin at the **marker
        center**. For a flat pad with the tag in the bottom-left, the pad center is
        offset by ``(pad_size - marker_length)/2`` in **world +x and +y** (same
        convention as :meth:`HanoiTask.populate_scene`, which parents tags under an
        **axis-aligned** pad box).

        We intentionally **do not** apply that offset using the measured marker
        rotation ``R``: a few degrees (or a near-π yaw ambiguity) would swing the
        ~9 cm in-plane offset and place pad centers **O(10–20 cm)** away from the
        mesh, while the Viser pad/tag geometry still uses identity pad rotation.
        """
        T = np.asarray(T_base_marker, dtype=float).reshape(4, 4)
        p = T[:3, 3].copy()
        ps = float(self.pad_size)
        tw = float(self.marker_length)
        d = (ps - tw) / 2
        p[0] += d
        p[1] += d
        p[2] = float(PAD_SLAB_CENTER_Z_M)
        return p

    def find_tag_poses(
        self,
        frame: np.ndarray,
        *,
        camera_matrix: np.ndarray | None = None,
        dist_coeffs: np.ndarray | None = None,
    ) -> list[tuple[int, np.ndarray]]:
        """Return ``[(tag_id, T_cam_marker), ...]`` for every ArUco tag in ``frame``.

        Note: this returns the *camera-frame* pose of each marker, not the base
        frame. Callers that need base-frame poses should multiply by the current
        camera-in-base transform (``T_base_cam @ T_cam_marker``).

        Optional ``camera_matrix`` / ``dist_coeffs`` are forwarded to
        :meth:`~me235b.detector.ArucoDetector.find_tags` for ``solvePnP`` and
        should match how ``frame`` was rasterized.
        """
        detections = self.find_tags(frame, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        results: list[tuple[int, np.ndarray]] = []
        for tag_id, rvec, tvec in detections:
            import cv2

            R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))
            T = np.eye(4, dtype=float)
            T[:3, :3] = R
            T[:3, 3] = np.asarray(tvec, dtype=float).reshape(3)
            results.append((int(tag_id), T))
        return results

    def grasp_pose_for(
        self,
        tag: BoxTag,
        T_base_aruco: np.ndarray,
        *,
        use_measured_tag_orientation: bool = False,
    ) -> np.ndarray:
        """Base -> gripper-center transform for grasping the block at ``T_base_aruco``.

        Returns ``Tbg`` as defined in Lab 3 Part 2 prep (a): "transform between
        the base of the robot and the center of the gripper in the open position
        such that it can be closed to grab the object."

        The block's ArUco tag sits in the bottom-left corner of the top face; the
        prism extends into the block's ``+x`` / ``+y`` / ``-z`` (below the tag).
        By default the orientation is fixed to :data:`GRASP_ORIENTATION` — we
        ignore the measured ArUco rotation because the PDF assumes blocks are
        base-frame aligned. With ``use_measured_tag_orientation=True``, the
        gripper z axis follows **−tag +z** and gripper x follows the tag's
        in-plane x (see :func:`grasp_orientation_from_measured_tag`).
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
        if use_measured_tag_orientation:
            T_grasp[:3, :3] = grasp_orientation_from_measured_tag(T_base_aruco)
        else:
            T_grasp[:3, :3] = GRASP_ORIENTATION
        T_grasp[:3, 3] = T_base_block[:3, 3]
        return T_grasp

    def place_pose_for(self, _tag: PadTag, T_base_aruco: np.ndarray) -> np.ndarray:
        """Base -> gripper-center transform for dropping a block onto the given pad.

        The tower is a 20 cm square with its ArUco tag in the bottom-left corner,
        lying flat at ``z ~ 0`` with base-frame-aligned orientation. ``T_base_aruco``
        is the fused **marker-center** pose from OpenCV. We convert to the pad
        center, then offset by ``block_thickness/2`` in ``z`` for the place height.
        """
        pad_c = self.pad_center_from_marker_pose(T_base_aruco)
        # Block center sits on the table plane (pad top ≈ 0); xy from pad center.
        place_pos = np.array(
            [float(pad_c[0]), float(pad_c[1]), self.block_thickness / 2],
            dtype=float,
        )

        T_place = np.eye(4, dtype=float)
        T_place[:3, :3] = GRASP_ORIENTATION
        T_place[:3, 3] = place_pos
        return T_place

    def grasp_dict(
        self,
        detections: dict[BoxTag, np.ndarray],
        *,
        use_measured_tag_orientation: bool = False,
    ) -> dict[BoxTag, np.ndarray]:
        return {
            tag: self.grasp_pose_for(tag, T, use_measured_tag_orientation=use_measured_tag_orientation)
            for tag, T in detections.items()
        }

    def place_dict(self, detections: dict[PadTag, np.ndarray]) -> dict[PadTag, np.ndarray]:
        return {tag: self.place_pose_for(tag, T) for tag, T in detections.items()}


def _default_home_pose() -> np.ndarray:
    """Home pose: 10 cm above the base's y = -0.5 region, gripper pointing down."""
    T = np.eye(4, dtype=float)
    T[:3, :3] = GRASP_ORIENTATION
    T[:3, 3] = [0.30, -0.40, 0.40]
    return T


# Lab 3 horizontal workspace (base frame, metres): where pads and tags may appear.
# Matches the scan tiling description in :func:`_default_scan_points` and the PDF.
TAG_WORKSPACE_X_MIN: float = -0.7
TAG_WORKSPACE_X_MAX: float = 0.7
TAG_WORKSPACE_Y_MIN: float = -1.0
TAG_WORKSPACE_Y_MAX: float = 0.0

# Default pad geometric centers (base frame, metres) for synthetic / random layouts.
DEFAULT_PAD_CENTERS_LAB3: dict[PadTag, np.ndarray] = {
    PadTag.START_PAD: np.array([-0.20, -0.85, 0.0], dtype=float),
    PadTag.MIDDLE_PAD: np.array([0.10, -0.85, 0.0], dtype=float),
    PadTag.END_PAD: np.array([0.30, -0.85, 0.0], dtype=float),
}


def _pad_centers_xy_overlap(
    c1: np.ndarray,
    c2: np.ndarray,
    pad_side: float,
    *,
    clearance: float = 0.02,
) -> bool:
    """True if two axis-aligned ``pad_side`` pads in the table plane would overlap (xy)."""
    lim = float(pad_side) + float(clearance)
    return bool(abs(float(c1[0]) - float(c2[0])) < lim and abs(float(c1[1]) - float(c2[1])) < lim)


def _sample_non_overlapping_pad_centers(
    rng: np.random.Generator,
    pad_side: float,
    *,
    max_trials: int = 800,
) -> dict[PadTag, np.ndarray]:
    """Independent random pad centers in the tag workspace with no xy overlap."""
    h = pad_side / 2
    x_lo = TAG_WORKSPACE_X_MIN + h
    x_hi = TAG_WORKSPACE_X_MAX - h
    y_lo = TAG_WORKSPACE_Y_MIN + h
    y_hi = TAG_WORKSPACE_Y_MAX - h
    tags = list(PadTag)
    for _ in range(max_trials):
        pts = {
            tag: np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi), 0.0], dtype=float)
            for tag in tags
        }
        ok = True
        for i, ti in enumerate(tags):
            for tj in tags[i + 1 :]:
                if _pad_centers_xy_overlap(pts[ti], pts[tj], pad_side):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return pts
    # Rare: keep the known-good default triangle (independent jitter could overlap).
    return {tag: DEFAULT_PAD_CENTERS_LAB3[tag].copy() for tag in tags}


def _scan_detections_complete(
    box_detections: dict[BoxTag, np.ndarray], pad_detections: dict[PadTag, np.ndarray]
) -> bool:
    """True iff every lab tag ID appears in the scan merge dicts."""
    return len(box_detections) == len(BoxTag) and len(pad_detections) == len(PadTag)


def _serpentine_order_scan_points(
    points: Sequence[np.ndarray] | np.ndarray,
    *,
    y_row_tol: float,
) -> list[np.ndarray]:
    """Order scan poses in alternating row direction (boustrophedon) in the table plane.

    Points are grouped into rows by similar *y* (within ``y_row_tol``), ordered along
    *x* left-to-right on even row indices and right-to-left on odd indices, so the
    path does not reset to the far *x* after each row.
    """
    if isinstance(points, np.ndarray) and (points.size == 0 or points.shape[0] == 0):
        return []
    if not isinstance(points, np.ndarray) and not points:
        return []
    p = np.stack([np.asarray(x, dtype=float).reshape(3) for x in points], axis=0)
    if p.shape[0] == 1:
        return [p[0]]
    ind = np.lexsort((p[:, 0], p[:, 1]))
    p = p[ind]
    rows: list[np.ndarray] = []
    i = 0
    n = p.shape[0]
    while i < n:
        y0 = p[i, 1]
        j = i + 1
        while j < n and abs(p[j, 1] - y0) <= y_row_tol:
            j += 1
        row = p[i:j]
        row = row[row[:, 0].argsort()]
        if len(rows) % 2 == 1:
            row = row[::-1]
        rows.append(row)
        i = j
    out = np.vstack(rows)
    return [out[k] for k in range(out.shape[0])]


def _create_point_grid(xmin: float, xmax: float, ymin: float, ymax: float, x_points: int, y_points: int, scan_height: float) -> np.ndarray:
    x_cell = (xmax - xmin) / x_points
    y_cell = (ymax - ymin) / y_points
    xs = np.arange(xmin, xmax + x_cell, x_cell)
    ys = np.arange(ymin, ymax + y_cell, y_cell)
    parts: list[np.ndarray] = []
    for ri, y in enumerate(ys):
        x_seq = xs if (ri % 2 == 0) else xs[::-1]
        for x in x_seq:
            parts.append(np.array([x, y, scan_height], dtype=float))
    return np.stack(parts, axis=0) if parts else np.zeros((0, 3), dtype=float)


def _default_scan_points(scan_height: float = DEFAULT_SCAN_HEIGHT) -> np.ndarray:
    """Six scan positions tiled across :data:`TAG_WORKSPACE_*` in the base frame."""
    # return np.array(
    #     [
    #         [-0.35, -0.33, scan_height],
    #         [0.0, -0.33, scan_height],
    #         [0.35, -0.33, scan_height],
    #         [-0.35, -0.67, scan_height],
    #         [0.0, -0.67, scan_height],
    #         [0.35, -0.67, scan_height],
    #     ],
    #     dtype=float,
    # )
    y_points = 3
    x_points = 3
    x_min = -0.5#TAG_WORKSPACE_X_MIN
    x_max = 0.5#TAG_WORKSPACE_X_MAX
    y_min = -0.7#TAG_WORKSPACE_Y_MIN
    y_max = -0.3#TAG_WORKSPACE_Y_MAX - 0.4
    return _create_point_grid(x_min, x_max, y_min, y_max, x_points, y_points, scan_height)


# Shared with :meth:`HanoiTask.scan` and :func:`compute_auto_scan_points` so scan
# motion matches offline feasibility checks.
_SCAN_CAMERA_MOUNT_IK: dict[str, float | bool | int] = {
    "position_only": False,
    "trans_weight": 20.0,
    "rot_weight": 1.0,
    "pos_tol": 1.0e-2,
    "rot_tol": 0.85,
    "max_iter": 120,
    "pre_solve_position": True,
}


def _bootstrap_theta_for_scan_planning(kin: UR10e, T_home: np.ndarray) -> np.ndarray:
    """Flange IK at ``T_home`` gives a joint seed in the same neighborhood as ``home()``."""
    T_home = np.asarray(T_home, dtype=float).reshape(4, 4)
    th, _, info = kin.ik(T_home, theta_seed=np.zeros(6, dtype=float))
    if info.get("success", False) and not np.any(np.isnan(th)):
        return np.asarray(th, dtype=float).reshape(6)
    return np.zeros(6, dtype=float)


def _scan_cell_index(
    x: float,
    y: float,
    *,
    x_min: float,
    y_min: float,
    cell: float,
    nx: int,
    ny: int,
) -> tuple[int, int]:
    ix = int((float(x) - float(x_min)) / float(cell))
    iy = int((float(y) - float(y_min)) / float(cell))
    return int(np.clip(ix, 0, nx - 1)), int(np.clip(iy, 0, ny - 1))


def _farthest_point_indices(xy: np.ndarray, k: int) -> list[int]:
    """Greedy max-min subsample of ``xy`` rows (2D points) down to ``k`` indices."""
    n = int(xy.shape[0])
    if k <= 0 or n == 0:
        return []
    if k >= n:
        return list(range(n))
    chosen = [0]
    dist = np.linalg.norm(xy - xy[0], axis=1)
    while len(chosen) < k:
        j = int(np.argmax(dist))
        if float(dist[j]) <= 1e-9:
            break
        chosen.append(j)
        dist = np.minimum(dist, np.linalg.norm(xy - xy[j], axis=1))
    return sorted(set(chosen))


def _chain_validate_scan_points(
    kin: UR10e,
    T5c: np.ndarray,
    points: list[np.ndarray],
    *,
    T_home: np.ndarray,
) -> list[np.ndarray]:
    """Keep poses that succeed under the same IK chaining used at scan runtime."""
    T_home = np.asarray(T_home, dtype=float).reshape(4, 4)
    T5c = np.asarray(T5c, dtype=float).reshape(4, 4)
    seed = _bootstrap_theta_for_scan_planning(kin, T_home)
    ok_list: list[np.ndarray] = []
    for p in points:
        T_bc = make_T_rpy(np.asarray(p, dtype=float).reshape(3), [np.pi, 0.0, 0.0])
        th, _, info = kin.ik_camera_mount(
            T_bc,
            T5c,
            theta_seed=seed,
            **_SCAN_CAMERA_MOUNT_IK,
        )
        if not info.get("success", False) or np.any(np.isnan(th)):
            continue
        seed = th
        ok_list.append(np.asarray(p, dtype=float).reshape(3))
    return ok_list


def compute_auto_scan_points(
    kin: UR10e,
    T5c: np.ndarray,
    *,
    T_home: np.ndarray | None = None,
    scan_height: float = DEFAULT_SCAN_HEIGHT,
    cell_size: float = 0.22,
    sample_step: float = 0.11,
    max_points: int = 0,
    verbose: bool = False,
) -> np.ndarray:
    """Pick base-frame camera positions over the tag workspace with feasible link-5 IK.

    Candidates lie on a regular ``(x, y)`` grid at ``z = scan_height`` with the lab
    top-down camera RPY target (``roll=π``). For each grid cell, the pose with
    the smallest IK rotation error is kept so the realized view stays as close
    to straight-down as the five controllable joints allow, while tiling ``xy``
    for coverage.

    Falls back toward :func:`_default_scan_points` if the grid yields too few poses.
    """
    T_home = _default_home_pose() if T_home is None else np.asarray(T_home, dtype=float).reshape(4, 4)
    T5c = np.asarray(T5c, dtype=float).reshape(4, 4)
    cell = max(0.05, float(cell_size))
    step = max(0.04, float(sample_step))
    if step > cell * 0.95:
        step = cell * 0.5

    x_min, x_max = float(TAG_WORKSPACE_X_MIN), float(TAG_WORKSPACE_X_MAX)
    y_min, y_max = float(TAG_WORKSPACE_Y_MIN), float(TAG_WORKSPACE_Y_MAX)
    nx = max(1, int(np.ceil((x_max - x_min) / cell)))
    ny = max(1, int(np.ceil((y_max - y_min) / cell)))

    xs = np.arange(x_min, x_max + 0.5 * step, step, dtype=float)
    ys = np.arange(y_min, y_max + 0.5 * step, step, dtype=float)
    candidates: list[tuple[float, float]] = [(float(x), float(y)) for y in ys for x in xs]

    seed = _bootstrap_theta_for_scan_planning(kin, T_home)
    # Deterministic candidate ordering for IK seeding (serpentine applied to output later).
    candidates.sort(key=lambda xy: (xy[1], xy[0]))

    successes: list[tuple[np.ndarray, float, float, tuple[int, int]]] = []
    for x, y in candidates:
        T_bc = make_T_rpy(np.array([x, y, float(scan_height)], dtype=float), [np.pi, 0.0, 0.0])
        th, _, info = kin.ik_camera_mount(
            T_bc,
            T5c,
            theta_seed=seed,
            **_SCAN_CAMERA_MOUNT_IK,
        )
        if not info.get("success", False) or np.any(np.isnan(th)):
            continue
        seed = th
        rot_e = float(info.get("rot_err_rad", 1.0e9))
        pos_e = float(info.get("pos_err_m", 1.0e9))
        cell_key = _scan_cell_index(x, y, x_min=x_min, y_min=y_min, cell=cell, nx=nx, ny=ny)
        successes.append((np.array([x, y, float(scan_height)], dtype=float), rot_e, pos_e, cell_key))

    best_by_cell: dict[tuple[int, int], tuple[np.ndarray, float, float]] = {}
    for xyz, rot_e, pos_e, ck in successes:
        prev = best_by_cell.get(ck)
        if prev is None or rot_e < prev[1] or (rot_e == prev[1] and pos_e < prev[2]):
            best_by_cell[ck] = (xyz, rot_e, pos_e)

    row_tol = max(float(step) * 0.51, 0.02)
    chosen: list[np.ndarray] = [t[0] for t in best_by_cell.values()]
    chosen = _serpentine_order_scan_points(chosen, y_row_tol=row_tol)
    chosen = _chain_validate_scan_points(kin, T5c, chosen, T_home=T_home)

    def _try_add_default() -> None:
        seed_d = _bootstrap_theta_for_scan_planning(kin, T_home)
        for p in _default_scan_points(scan_height):
            x, y = float(p[0]), float(p[1])
            T_bc = make_T_rpy(np.asarray(p, dtype=float).reshape(3), [np.pi, 0.0, 0.0])
            th, _, info = kin.ik_camera_mount(
                T_bc,
                T5c,
                theta_seed=seed_d,
                **_SCAN_CAMERA_MOUNT_IK,
            )
            if not info.get("success", False) or np.any(np.isnan(th)):
                continue
            seed_d = th
            ck = _scan_cell_index(x, y, x_min=x_min, y_min=y_min, cell=cell, nx=nx, ny=ny)
            if ck not in best_by_cell:
                best_by_cell[ck] = (np.asarray(p, dtype=float).reshape(3), float(info.get("rot_err_rad", 0.0)), 0.0)

    if len(chosen) < 4:
        _try_add_default()
        chosen = [t[0] for t in best_by_cell.values()]
        chosen = _serpentine_order_scan_points(chosen, y_row_tol=row_tol)
        chosen = _chain_validate_scan_points(kin, T5c, chosen, T_home=T_home)

    if len(chosen) == 0:
        if verbose:
            print("[hanoi] auto scan points: no feasible poses; using default scan grid.")
        return _default_scan_points(scan_height)

    max_pts = int(max_points)
    if max_pts > 0 and len(chosen) > max_pts:
        xy = np.array([[float(p[0]), float(p[1])] for p in chosen], dtype=float)
        keep = _farthest_point_indices(xy, max_pts)
        chosen = [chosen[i] for i in keep]
        chosen = _serpentine_order_scan_points(chosen, y_row_tol=row_tol)
        chosen = _chain_validate_scan_points(kin, T5c, chosen, T_home=T_home)

    out = np.stack(chosen, axis=0)
    if verbose:
        print(
            f"[hanoi] auto scan points: {out.shape[0]} poses "
            f"(workspace cells {nx}x{ny}, step={step:.3f} m, cell={cell:.3f} m)."
        )
    return out


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
        scan_settle_s: float = 0.0,
        synthetic_scan_detection: bool = False,
        tag_grasp_orientation: bool = False,
        sim_vision_gt_tolerance_m: float | None = None,
    ) -> None:
        self.controller = controller
        self.detector = detector
        self.camera = camera
        self.T_home = _default_home_pose() if T_home is None else np.asarray(T_home, dtype=float)
        self.T5c = T5C_DEFAULT if T5c is None else np.asarray(T5c, dtype=float)
        self.scan_points = _default_scan_points() if scan_points is None else np.asarray(scan_points, dtype=float)
        self.approach_height = float(approach_height)
        self.scan_settle_s = float(scan_settle_s)
        self.synthetic_scan_detection = bool(synthetic_scan_detection)
        self.tag_grasp_orientation = bool(tag_grasp_orientation)
        self._sim_vision_gt_tolerance_m = (
            None if sim_vision_gt_tolerance_m is None else float(sim_vision_gt_tolerance_m)
        )
        self._sim_gt_pad_centers: dict[PadTag, np.ndarray] | None = None
        self._sim_gt_block_centers: dict[BoxTag, np.ndarray] | None = None
        self._demo_layout_cp: tuple[
            dict[PadTag, np.ndarray],
            dict[PadTag, list[BoxTag]],
            dict[BoxTag, np.ndarray],
            dict[BoxTag, np.ndarray],
        ] | None = None

        self._box_handles: dict[BoxTag, str] = {}
        self._pad_handles: dict[PadTag, str] = {}
        self._holding: BoxTag | None = None

        # Internal world model.
        # pad_centers: pad ID -> (x, y, z) of the pad's geometric center in base frame.
        # pad_stacks:  pad ID -> list of blocks on that pad, bottom -> top.
        # block_centers: block ID -> (x, y, z) of the block's geometric center in base frame.
        #   After a scan, xy is taken from the ArUco detection (so corner-stacked blocks
        #   are handled correctly) and z is derived from the block's stack position
        #   (robust against the ArUco z-noise the PDF warns about).
        #   After a move, we update this entry to the pad-centered placement pose.
        self.pad_centers: dict[PadTag, np.ndarray] = {}
        self.pad_stacks: dict[PadTag, list[BoxTag]] = {p: [] for p in PadTag}
        self.block_centers: dict[BoxTag, np.ndarray] = {}
        # When :attr:`tag_grasp_orientation` is True: last 3x3 gripper rotation (base frame)
        # per block, from the tag at scan time; reset to :data:`GRASP_ORIENTATION`
        # after each axis-aligned :meth:`place_pose_for_top` placement.
        self._block_grasp_R: dict[BoxTag, np.ndarray] = {}

    @property
    def renderer(self) -> SimulationRenderer:
        return self.controller.renderer

    def home(self) -> dict:
        """Drive the gripper to the configured home pose."""
        return self.controller.home(self.T_home, verify_fk=True)

    def _strict_vision_sim(self) -> bool:
        """Simulate with Viser synthetic frames only (vision-first / IRL parity path)."""
        return bool(
            self.controller.simulate
            and self.camera is None
            and self.synthetic_scan_detection
        )

    def _raise_if_sim_vision_gt_exceeds_tolerance(self) -> None:
        """Sim oracle: fail when ingested centers drift from last Randomize layout.

        This is **not** rigid-body physics (no contacts, slip, or dynamics). It is an
        optional consistency check that synthetic vision + fusion recovered the same
        pad/block centers (base frame) that :meth:`randomize_sim_layout` established.
        """
        tol = self._sim_vision_gt_tolerance_m
        if tol is None or tol <= 0.0:
            return
        gt_p = self._sim_gt_pad_centers
        gt_b = self._sim_gt_block_centers
        if gt_p is None or gt_b is None:
            return
        if any(p not in gt_p for p in PadTag) or any(b not in gt_b for b in BoxTag):
            return

        worst = 0.0
        worst_name = ""
        for p in PadTag:
            if p not in self.pad_centers:
                return
            e = float(np.linalg.norm(self.pad_centers[p] - gt_p[p]))
            if e > worst:
                worst, worst_name = e, f"pad {p.name}"
        for b in BoxTag:
            if b not in self.block_centers:
                return
            e = float(np.linalg.norm(self.block_centers[b] - gt_b[b]))
            if e > worst:
                worst, worst_name = e, f"block {b.name}"
        if worst > tol:
            raise ValueError(
                f"Sim vision vs Randomize layout: max center error {worst:.4f} m at {worst_name} "
                f"(tolerance {tol} m). Tighten scan / intrinsics or increase tolerance."
            )

    def _reset_planner_state_only(self) -> None:
        """Clear pad/block world model without removing Viser meshes (used before re-scan)."""
        self.pad_centers.clear()
        self.pad_stacks = {p: [] for p in PadTag}
        self.block_centers.clear()
        self._block_grasp_R.clear()
        self._holding = None

    def _clear_scene_and_planner_for_strict_scan(self) -> None:
        """Empty the scene and planner so the first successful scan defines everything."""
        self.renderer.clear_hanoi_scene_objects()
        self._box_handles.clear()
        self._pad_handles.clear()
        self._reset_planner_state_only()

    def _require_full_world_model(self) -> None:
        """Raise if any lab tag is missing from the world model (hardware and strict sim)."""
        missing_pads = [p for p in PadTag if p not in self.pad_centers]
        missing_boxes = [b for b in BoxTag if b not in self.block_centers]
        if missing_pads or missing_boxes:
            missing = [t.name for t in (*missing_pads, *missing_boxes)]
            raise ValueError(
                f"HanoiTask.run: scan did not find all required tags (missing: {missing}). "
                "Check camera exposure and scan point coverage."
            )

    def _grab_frame(self, *, flush: int = 3) -> np.ndarray | None:
        """Read a frame from the camera, flushing a few stale buffered frames first.

        USB webcams often keep a 2-3 frame internal buffer, so the first ``read()``
        after a robot motion returns an image from before the move finished. We
        discard ``flush`` frames before returning the one used for detection.
        """
        if self.camera is None:
            return None
        for _ in range(max(0, flush)):
            self.camera.read()
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
        then update the internal world model (pad centers + stacks).

        After each successful move, :attr:`scan_settle_s` seconds of stillness
        elapse before reading the camera or refreshing scan snapshots (helps
        rolling-shutter webcams and motion blur).

        If ``self.camera`` is ``None`` and :attr:`synthetic_scan_detection` is
        false, only motions run; returned detections are empty and
        :meth:`_ingest_detections` is not called (caller supplies poses).

        If ``synthetic_scan_detection`` is true (Viser path), a synthetic
        frame is rendered at lab image size (``LAB3_CAMERA_IMAGE_W`` × ``LAB3_CAMERA_IMAGE_H``) with the renderer's **K** and **D**
        (lab PDF calibration, or ideal pinhole if Viser was built with
        ``sim_camera_intrinsics="simple"``), matching
        :meth:`me235b.sim.ViserRenderer.synthetic_aruco_intrinsics`.
        A complete tag set is ingested; otherwise the prior model is left unchanged
        (unless :meth:`HanoiTask._run_body` strict vision mode cleared it first).

        Repeated sightings of the same tag across scan poses are merged with
        :func:`me235b.transforms.fuse_rigid_transforms` (MAD outlier rejection on
        translation norms, then on rotation tangent norms). Viser draws one triad
        per fused **marker** pose (full 4×4: tag origin on the pad corner or block
        face). The internal world model from :meth:`_ingest_detections` still uses
        pad centers and stack-based block ``z`` as in the lab spec.
        """
        box_samples: defaultdict[BoxTag, list[np.ndarray]] = defaultdict(list)
        pad_samples: defaultdict[PadTag, list[np.ndarray]] = defaultdict(list)

        self.renderer.clear_scan_pose_estimates()

        n_scan = len(self.scan_points)
        for scan_index, scan_point in enumerate(self.scan_points):
            T_bc = make_T_rpy(scan_point, [np.pi, 0.0, 0.0])
            # Weighted pose after position pre-solve: link-5 has 5 DOF, so full RPY is
            # not reachable; SciPy refines toward downward view while holding the grid
            # point (see :meth:`me235b.kinematics.UR10e.ik_camera_mount`).
            ok, _theta, info = self.controller.move_to_camera_pose(
                T_bc,
                self.T5c,
                **_SCAN_CAMERA_MOUNT_IK,
            )
            if not ok:
                if self.controller.verbose:
                    print(f"[hanoi] scan move IK failed at {scan_point}: {info.get('message','')}")
                continue

            if self.scan_settle_s > 0.0:
                time.sleep(self.scan_settle_s)

            frame = self._grab_frame() if self.camera is not None else None
            if frame is None and self.camera is None and self.synthetic_scan_detection:
                synth = getattr(self.renderer, "synthetic_camera_bgr_for_pose", None)
                if callable(synth):
                    frame = synth(self.controller.current_q_class)

            self.renderer.on_scan_snapshot(
                scan_index,
                total=n_scan,
                q=self.controller.current_q_class,
                frame_bgr=frame,
            )

            if frame is None:
                continue

            T_base_cam = self._current_T_base_cam()

            # solvePnP must use the same (K, D) as the synthetic raster (Viser).
            pnp_K: np.ndarray | None = None
            pnp_D: np.ndarray | None = None
            if self.camera is None and self.synthetic_scan_detection:
                intr_fn = getattr(self.renderer, "synthetic_aruco_intrinsics", None)
                if intr_fn is not None:
                    pair = intr_fn()
                    if pair is not None:
                        pnp_K, pnp_D = pair
            tag_iter = self.detector.find_tag_poses(
                frame,
                camera_matrix=pnp_K,
                dist_coeffs=pnp_D,
            )

            for tag_id, T_cam_marker in tag_iter:
                T_base_marker = T_base_cam @ T_cam_marker

                try:
                    box = BoxTag(tag_id)
                    box_samples[box].append(T_base_marker)
                    continue
                except ValueError:
                    pass
                try:
                    pad = PadTag(tag_id)
                    pad_samples[pad].append(T_base_marker)
                    continue
                except ValueError:
                    pass
                if self.controller.verbose:
                    print(f"[hanoi] ignoring non-Hanoi tag id={tag_id}")

            self.renderer.on_demo_checkpoint()

        box_detections: dict[BoxTag, np.ndarray] = {
            tag: fuse_rigid_transforms(samples) for tag, samples in box_samples.items()
        }
        pad_detections: dict[PadTag, np.ndarray] = {
            tag: fuse_rigid_transforms(samples) for tag, samples in pad_samples.items()
        }

        ingest = False
        if self.camera is not None:
            ingest = True
        elif (
            self.controller.simulate
            and self.synthetic_scan_detection
            and _scan_detections_complete(box_detections, pad_detections)
        ):
            ingest = True
        elif (
            self.controller.simulate
            and self.synthetic_scan_detection
            and self.controller.verbose
            and (box_detections or pad_detections)
            and not _scan_detections_complete(box_detections, pad_detections)
        ):
            print(
                "[hanoi] synthetic camera scan: incomplete tag set at this merge; "
                "keeping prior world model unless a later scan pose fills all IDs."
            )

        if ingest:
            self._ingest_detections(box_detections, pad_detections)
            self._raise_if_sim_vision_gt_exceeds_tolerance()
        elif (
            self.controller.simulate
            and self.synthetic_scan_detection
            and self.camera is None
            and not (box_detections or pad_detections)
            and self.controller.verbose
        ):
            print(
                "[hanoi] synthetic camera scan: no tags detected in rendered views "
                "(check Viser scene / intrinsics)."
            )

        for name, T in self._scan_plot_transforms(
            box_detections,
            pad_detections,
            snap_translation_to_mesh=ingest,
        ):
            self.renderer.add_scan_pose_estimate(name, T)

        return box_detections, pad_detections

    def _mesh_marker_center_world_pad(self, pad: PadTag) -> np.ndarray:
        """World marker-center position matching :meth:`populate_scene` pad + tag."""
        pc = self.pad_centers[pad]
        ps = float(self.detector.pad_size)
        tw = float(self.detector.marker_length)
        d = (ps - tw) / 2
        return np.array(
            [
                float(pc[0] - d),
                float(pc[1] - d),
                float(PAD_SLAB_CENTER_Z_M + PAD_TAG_LOCAL_Z_M),
            ],
            dtype=float,
        )

    def _mesh_marker_center_world_box(self, box: BoxTag) -> np.ndarray:
        """World marker-center position matching :meth:`populate_scene` box + tag."""
        c = self.block_centers[box]
        w = float(BOX_WIDTHS[box])
        tw = float(self.detector.marker_length)
        bt = float(self.detector.block_thickness)
        d = (w - tw) / 2
        return np.array(
            [
                float(c[0] - d),
                float(c[1] - d),
                float(c[2] + bt / 2 + BLOCK_TAG_Z_EPSILON_M),
            ],
            dtype=float,
        )

    def _scan_plot_transforms(
        self,
        box_detections: dict[BoxTag, np.ndarray],
        pad_detections: dict[PadTag, np.ndarray],
        *,
        snap_translation_to_mesh: bool,
    ) -> list[tuple[str, np.ndarray]]:
        """World-frame fused marker **rotation** with optional mesh-aligned **origin**.

        After a full ingest, triad translations match :meth:`populate_scene` tag
        placement (axis-aligned pad/box + local tag offsets) so Viser frames sit
        on the same marker centers as the textured quads. Rotation still comes from
        :func:`fuse_rigid_transforms` (PnP) so tilt/yaw error remains visible.
        """
        out: list[tuple[str, np.ndarray]] = []
        for tag, T_aruco in pad_detections.items():
            Tp = np.asarray(T_aruco, dtype=float).copy()
            if snap_translation_to_mesh and tag in self.pad_centers:
                Tp[:3, 3] = self._mesh_marker_center_world_pad(tag)
            out.append((f"fused_tag_pad_{int(tag)}", Tp))
        for box, T_aruco in box_detections.items():
            Tp = np.asarray(T_aruco, dtype=float).copy()
            if snap_translation_to_mesh and box in self.block_centers:
                Tp[:3, 3] = self._mesh_marker_center_world_box(box)
            out.append((f"fused_tag_box_{int(box)}", Tp))
        return out

    def _ingest_detections(
        self,
        box_detections: dict[BoxTag, np.ndarray],
        pad_detections: dict[PadTag, np.ndarray],
    ) -> None:
        """Populate pad_centers, pad_stacks, and block_centers from a scan.

        Pad centers come from :meth:`HanoiDetector.pad_center_from_marker_pose`
        (marker-center detections + axis-aligned pad offset in world ``xy``, slab
        ``z`` matching :meth:`populate_scene`). Each block is assigned to whichever
        pad center it's closest to in xy.

        Per the PDF, blocks in the starting tower are *corner-stacked* (not centered
        on the pad) so their tags stay visible. We therefore use the ArUco's
        detected xy for each block's position instead of assuming pad-centered
        placement. The block's z is derived from its size-sorted stack position
        (the PDF explicitly warns about ArUco z-estimation noise and suggests
        falling back to the known block thicknesses).
        """
        self.pad_centers.clear()
        for tag, T_aruco in pad_detections.items():
            self.pad_centers[tag] = self.detector.pad_center_from_marker_pose(T_aruco)

        self.pad_stacks = {p: [] for p in PadTag}
        self.block_centers.clear()
        self._block_grasp_R.clear()

        if not self.pad_centers:
            return

        # First pass: compute each block's base-frame xy from its ArUco detection
        # and assign the block to the pad whose center it is xy-closest to.
        block_xy: dict[BoxTag, np.ndarray] = {}
        for box, T_aruco in box_detections.items():
            T_block = self.detector.grasp_pose_for(
                box,
                T_aruco,
                use_measured_tag_orientation=self.tag_grasp_orientation,
            )
            block_xy[box] = T_block[:3, 3][:2].copy()
            if self.tag_grasp_orientation:
                self._block_grasp_R[box] = grasp_orientation_from_measured_tag(T_aruco)

            closest_pad = min(
                self.pad_centers,
                key=lambda p: float(np.linalg.norm(self.pad_centers[p][:2] - block_xy[box])),
            )
            self.pad_stacks[closest_pad].append(box)

        # Largest block on the bottom (Hanoi invariant).
        for stack in self.pad_stacks.values():
            stack.sort(key=lambda b: -BOX_WIDTHS[b])

        # Second pass: compute full block centers. xy = detected, z = size-sorted
        # stack position * block_thickness + block_thickness / 2.
        bt = self.detector.block_thickness
        for stack in self.pad_stacks.values():
            for i, box in enumerate(stack):
                xy = block_xy[box]
                self.block_centers[box] = np.array(
                    [float(xy[0]), float(xy[1]), i * bt + bt / 2],
                    dtype=float,
                )

    def _pose_from_center(self, center_xyz: np.ndarray, *, R: np.ndarray | None = None) -> np.ndarray:
        """Build a gripper pose targeting ``center_xyz`` with rotation ``R`` (default PDF grasp)."""
        T = np.eye(4, dtype=float)
        T[:3, :3] = GRASP_ORIENTATION if R is None else np.asarray(R, dtype=float)
        T[:3, 3] = np.asarray(center_xyz, dtype=float).reshape(3)
        return T

    def grasp_pose_for_top(self, pad: PadTag) -> tuple[BoxTag, np.ndarray]:
        """Return ``(block_on_top, gripper_pose)`` for the current top of ``pad``.

        Uses the block's last-known base-frame center (from the initial scan, updated
        after each successful place) so corner-stacked starting blocks are still
        grasped at their true xy. If :attr:`tag_grasp_orientation` is enabled, the
        gripper rotation comes from :attr:`_block_grasp_R` (from the tag at scan).
        """
        stack = self.pad_stacks.get(pad, [])
        if not stack:
            raise RuntimeError(f"grasp_pose_for_top: pad {pad.name} is empty.")
        top_box = stack[-1]
        if top_box not in self.block_centers:
            raise RuntimeError(f"grasp_pose_for_top: no known center for {top_box.name}.")
        R = None
        if self.tag_grasp_orientation:
            R = self._block_grasp_R.get(top_box)
        return top_box, self._pose_from_center(self.block_centers[top_box], R=R)

    def place_pose_for_top(self, pad: PadTag) -> np.ndarray:
        """Return the gripper pose for landing the next block centrally on ``pad``.

        We deliberately center the block on the destination pad (ignoring how the
        starting stack was corner-offset) because after the first move everything
        is in our hands and central stacking simplifies IK reachability and
        subsequent grasps.
        """
        if pad not in self.pad_centers:
            raise RuntimeError(f"place_pose_for_top: pad {pad.name} has no known center (run scan first).")
        bt = self.detector.block_thickness
        center_xy = self.pad_centers[pad]
        z = len(self.pad_stacks.get(pad, [])) * bt + bt / 2
        return self._pose_from_center([float(center_xy[0]), float(center_xy[1]), float(z)])

    def _hover_pose(self, target: np.ndarray) -> np.ndarray:
        """Return ``target`` shifted up by :attr:`approach_height` in the base z axis."""
        out = np.asarray(target, dtype=float).copy()
        out[2, 3] += self.approach_height
        return out

    def pick_up(self, tag: BoxTag, grasp_pose: np.ndarray) -> bool:
        """Open -> hover -> descend -> close -> lift sequence for ``tag`` at ``grasp_pose``.

        Opens the gripper first so a previously-closed gripper doesn't slam its
        fingers into the target block on descent.
        Returns ``True`` if every motion step's IK succeeded.
        """
        self.controller.gripper_open()
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
        """Render every known pad and every block at its detected position.

        Uses :attr:`block_centers` so corner-stacked starting blocks are drawn
        at their true xy, matching where the real gripper will grasp. Each
        object's ArUco tag is added as a child of its host scene node (so it
        moves rigidly with a grasped block) and positioned in the bottom-left
        corner of the host's top face, per the PDF convention.
        """
        self.renderer.clear_hanoi_scene_objects()
        self._box_handles.clear()
        self._pad_handles.clear()

        tw = self.detector.marker_length
        pad_size = self.detector.pad_size
        bt = self.detector.block_thickness

        for pad, center_xy in self.pad_centers.items():
            T_visual = np.eye(4, dtype=float)
            T_visual[:3, 3] = [float(center_xy[0]), float(center_xy[1]), PAD_SLAB_CENTER_Z_M]
            handle = self.renderer.add_box(
                T_visual,
                dimensions=(pad_size, pad_size, PAD_SLAB_THICKNESS_M),
                label=f"pad_{pad.name.lower()}",
                color=PAD_COLORS[pad],
            )
            self._pad_handles[pad] = handle

            # Pad's ArUco tag sits in the bottom-left corner of the 20 cm pad,
            # flush with the pad's top face. In the pad's local frame, bottom-left
            # is -x, -y relative to the pad center.
            T_tag = np.eye(4, dtype=float)
            T_tag[:3, 3] = [-(pad_size - tw) / 2, -(pad_size - tw) / 2, PAD_TAG_LOCAL_Z_M]
            self.renderer.add_aruco_tag(int(pad), T_tag, tw, parent=handle)

        for box, center in self.block_centers.items():
            width = BOX_WIDTHS[box]
            T_visual = np.eye(4, dtype=float)
            T_visual[:3, 3] = center
            handle = self.renderer.add_box(
                T_visual,
                dimensions=(width, width, bt),
                label=f"box_{box.name.lower()}",
                color=BOX_COLORS[box],
            )
            self._box_handles[box] = handle

            # Block's ArUco tag is in the bottom-left corner of its top face.
            # Local frame of the block: origin at block center, +z up toward top face.
            T_tag = np.eye(4, dtype=float)
            T_tag[:3, 3] = [
                -(width - tw) / 2,
                -(width - tw) / 2,
                bt / 2 + BLOCK_TAG_Z_EPSILON_M,
            ]
            self.renderer.add_aruco_tag(int(box), T_tag, tw, parent=handle)

    def randomize_sim_layout(self) -> None:
        """Pick random non-overlapping pad positions, then stack blocks on the start pad.

        Only meaningful in simulation (used by the Viser interactive demo).
        """
        rng = np.random.default_rng()
        pad_side = float(self.detector.pad_size)
        pad_centers = _sample_non_overlapping_pad_centers(rng, pad_side)

        ps = float(self.detector.pad_size)
        tw = float(self.detector.marker_length)
        R_id = np.eye(3, dtype=float)
        pads = {
            tag: make_T(
                np.asarray(pad_centers[tag], dtype=float)
                + np.array([-(ps - tw) / 2, -(ps - tw) / 2, 0.0], dtype=float),
                R_id,
            )
            for tag in PadTag
        }

        bt = self.detector.block_thickness
        tw = self.detector.marker_length
        corner_dxy = 0.015
        start_xy = pad_centers[PadTag.START_PAD][:2]
        stack_order = [BoxTag.LARGE_BOX, BoxTag.MEDIUM_BOX, BoxTag.SMALL_BOX]
        boxes: dict[BoxTag, np.ndarray] = {}
        for i, tag in enumerate(stack_order):
            w = BOX_WIDTHS[tag]
            center = np.array(
                [start_xy[0] + i * corner_dxy, start_xy[1] + i * corner_dxy, i * bt + bt / 2],
                dtype=float,
            )
            local = np.array([w / 2, tw / 2, -bt / 2], dtype=float)
            aruco_pos = center - R_id @ local
            boxes[tag] = make_T(aruco_pos, R_id)

        self._ingest_detections(boxes, pads)
        if self._sim_vision_gt_tolerance_m is not None and self._sim_vision_gt_tolerance_m > 0.0:
            self._sim_gt_pad_centers = {k: v.copy() for k, v in self.pad_centers.items()}
            self._sim_gt_block_centers = {k: v.copy() for k, v in self.block_centers.items()}
        self.populate_scene()
        self.renderer.clear_scan_pose_estimates()

    def save_demo_layout_checkpoint(self) -> None:
        self._demo_layout_cp = (
            {k: v.copy() for k, v in self.pad_centers.items()},
            {k: list(v) for k, v in self.pad_stacks.items()},
            {k: v.copy() for k, v in self.block_centers.items()},
            {k: v.copy() for k, v in self._block_grasp_R.items()},
        )

    def restore_demo_layout_checkpoint(self) -> None:
        if self._demo_layout_cp is None:
            return
        if len(self._demo_layout_cp) == 4:
            pc, ps, bc, bgr = self._demo_layout_cp
            self._block_grasp_R = {k: v.copy() for k, v in bgr.items()}
        else:
            pc, ps, bc = self._demo_layout_cp[:3]  # legacy 3-tuple checkpoints
            self._block_grasp_R.clear()
        self.pad_centers = {k: v.copy() for k, v in pc.items()}
        self.pad_stacks = {k: list(v) for k, v in ps.items()}
        self.block_centers = {k: v.copy() for k, v in bc.items()}
        self._holding = None

    def _fake_detections(self) -> tuple[dict[BoxTag, np.ndarray], dict[PadTag, np.ndarray]]:
        """Synthetic detections matching the PDF's Tower-of-Hanoi starting state.

        All three blocks start stacked on ``START_PAD`` (middle and end empty).
        Per the PDF, blocks are corner-stacked rather than centered so their
        ArUco tags remain visible: each upper block sits in the "top-right
        corner" of the block below it. We simulate that offset to exercise the
        corner-aware grasp math.
        """
        bt = self.detector.block_thickness
        tw = self.detector.marker_length
        R_id = np.eye(3, dtype=float)

        # Pad ArUco corners (bottom-left corner in each pad's 20 cm square).
        ps = float(self.detector.pad_size)
        tw = float(self.detector.marker_length)
        pad_centers = {tag: v.copy() for tag, v in DEFAULT_PAD_CENTERS_LAB3.items()}
        pads = {
            tag: make_T(
                np.asarray(center, dtype=float) + np.array([-(ps - tw) / 2, -(ps - tw) / 2, 0.0], dtype=float),
                R_id,
            )
            for tag, center in pad_centers.items()
        }

        # Stack all three blocks on START_PAD, largest on bottom. Corner offset
        # between adjacent blocks chosen so the smaller block's tag sits inside
        # the footprint of the larger block below but near its top-right corner.
        corner_dxy = 0.015  # 1.5 cm offset between adjacent blocks' centers, per lab setup
        start_xy = pad_centers[PadTag.START_PAD][:2]
        stack_order = [BoxTag.LARGE_BOX, BoxTag.MEDIUM_BOX, BoxTag.SMALL_BOX]
        boxes: dict[BoxTag, np.ndarray] = {}
        for i, tag in enumerate(stack_order):
            w = BOX_WIDTHS[tag]
            # Block center = pad xy + accumulated corner offset, z = stack height.
            center = np.array(
                [start_xy[0] + i * corner_dxy, start_xy[1] + i * corner_dxy, i * bt + bt / 2],
                dtype=float,
            )
            # Back out the ArUco position: grasp_pose_for applies R @ (w/2, tw/2, -bt/2).
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
        #self.home()
        if not self.place_down(dst, place_pose):
            return False
        #self.home()

        # Update the world model. After a place the block lives at the
        # destination pad's center at the new stack height, not wherever the
        # detection originally put it.
        self.pad_stacks[src].pop()
        self.pad_stacks[dst].append(top_box)
        self.block_centers[top_box] = place_pose[:3, 3].copy()
        if self.tag_grasp_orientation:
            # :meth:`place_pose_for_top` uses axis-aligned PDF grasp; model the block likewise.
            self._block_grasp_R[top_box] = GRASP_ORIENTATION.copy()
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

    def run(self, *, skip_sim_auto_seed: bool = False) -> None:
        """``home -> open gripper -> scan -> populate_scene -> solve -> home``."""
        self._run_body(skip_sim_auto_seed=skip_sim_auto_seed)

    def _run_body(self, *, skip_sim_auto_seed: bool) -> None:
        """Implementation of :meth:`run`.

        ``skip_sim_auto_seed`` (Viser interactive mode): skip the built-in
        PDF seed; the scene is already laid out via :meth:`randomize_sim_layout`.
        """
        self.home()
        # Guarantee a known gripper state before any grasping; a previously closed
        # gripper would otherwise slam its fingers into the first block.
        self.controller.gripper_open()

        self.renderer.set_camera_scan_path(self.scan_points)
        self.renderer.set_tag_workspace_outline(
            TAG_WORKSPACE_X_MIN,
            TAG_WORKSPACE_X_MAX,
            TAG_WORKSPACE_Y_MIN,
            TAG_WORKSPACE_Y_MAX,
        )

        # Viser + sim without synthetic vision cannot match the hardware vision path.
        if (
            self.controller.simulate
            and self.camera is None
            and isinstance(self.renderer, ViserRenderer)
            and not self.synthetic_scan_detection
        ):
            raise RuntimeError(
                "Lab 3 simulate + Viser requires synthetic vision (default on with Viser). "
                "Use --render none for offline PDF seeding without vision parity, or enable "
                "synthetic scan detection."
            )

        pre_populated = False

        if self._strict_vision_sim():
            if not skip_sim_auto_seed:
                self._clear_scene_and_planner_for_strict_scan()
            else:
                # Interactive: meshes already placed by randomize; re-scan defines the planner.
                self._reset_planner_state_only()
            self.scan()
            self._require_full_world_model()
            self.populate_scene()
            pre_populated = True
        elif self.camera is None and self.controller.simulate:
            if self.controller.verbose:
                print(
                    "[hanoi] simulate without Viser synthetic vision: using PDF seed; "
                    "IRL parity is not guaranteed."
                )
            if not skip_sim_auto_seed:
                if self.controller.verbose:
                    print(
                        "[hanoi] no camera + simulate=True -> seed scene from PDF layout, "
                        "then scan motion."
                    )
                box_detections, pad_detections = self._fake_detections()
                self._ingest_detections(box_detections, pad_detections)
                self.populate_scene()
                pre_populated = True
            else:
                pre_populated = True
            box_out, pad_out = self.scan()
            if self.synthetic_scan_detection and _scan_detections_complete(box_out, pad_out):
                self.populate_scene()
        else:
            self.scan()

        if not self.controller.simulate:
            self._require_full_world_model()

        if not pre_populated:
            self.populate_scene()
        self.solve_tower_of_hanoi()
        self.home()


class RenderMode(str, Enum):
    """Rendering backend for ``run_hanoi``."""

    none = "none"
    viser = "viser"


class ViserSimCamera(str, Enum):
    """Synthetic on-arm camera intrinsics (Viser + simulate only)."""

    lab = "lab"
    simple = "simple"


def _build_renderer(
    mode: RenderMode,
    kin: UR10e,
    *,
    viser_show_live_camera: bool = True,
    viser_sim_camera: ViserSimCamera = ViserSimCamera.lab,
    viser_scan_gallery_width: int = 200,
) -> SimulationRenderer | None:
    if mode is RenderMode.none:
        return None
    if mode is RenderMode.viser:
        from .sim import ViserRenderer

        return ViserRenderer(
            kin,
            show_camera_view=viser_show_live_camera,
            sim_camera_intrinsics=viser_sim_camera.value,
            scan_gallery_thumb_width=max(1, int(viser_scan_gallery_width)),
        )
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
    scan_settle_s: Annotated[
        float,
        typer.Option(
            help="Seconds to wait at each scan pose after the arm stops before "
            "grabbing a frame (reduces rolling-shutter / motion blur on USB cameras). "
            "Use 0 to skip.",
        ),
    ] = 0.25,
    synthetic_scan_detection: Annotated[
        bool,
        typer.Option(
            help="With --simulate and Viser, required for vision parity: ArUco on "
            "synthetic frames (renderer K/D); scene/plan come only from a full scan.",
        ),
    ] = True,
    viser_sim_camera: Annotated[
        ViserSimCamera,
        typer.Option(
            help="Viser synthetic camera only: 'lab' = PDF K/D (hardware parity); "
            "'simple' = ideal pinhole (~58° HFOV, D=0) for cleaner sim / camera-agnostic "
            "localization debugging. Real runs (--no-simulate) still use the detector's lab K/D.",
        ),
    ] = ViserSimCamera.lab,
    auto_scan_points: Annotated[
        bool,
        typer.Option(
            "--auto-scan-points/--no-auto-scan-points",
            help="Plan scan positions with link-5 camera IK over the tag workspace: "
            "cover xy in cells, keep the most downward-feasible pose per cell, then "
            "follow that path instead of the fixed lab grid.",
        ),
    ] = False,
    interactive_sim: Annotated[
        bool,
        typer.Option(
            help="With --simulate and Viser, use setup controls (randomize, start), "
            "pause/resume, and restart between runs. Ignored on real hardware.",
        ),
    ] = True,
    tag_grasp_orientation: Annotated[
        bool,
        typer.Option(
            help="Derive grasp orientation from each block's ArUco pose (yaw/tilt); "
            "default uses fixed base-aligned grasp from the PDF.",
        ),
    ] = False,
    viser_live_camera: Annotated[
        bool,
        typer.Option(
            "--viser-live-camera/--no-viser-live-camera",
            help="With Viser, refresh the on-arm camera GUI (and frustum texture) on every "
            "animation sub-step. Turn off to keep full synthetic rasterization only at scan "
            "snapshots (faster motion preview).",
        ),
    ] = True,
    viser_scan_gallery_width: Annotated[
        int,
        typer.Option(
            help="Viser 'Scan i/n' gallery image width (pixels); height follows the lab frame aspect "
            "(``LAB3_CAMERA_IMAGE_W``/``H`` in simple sim, scaled PDF calibration in lab mode). Default "
            "matches ``LAB3_CAMERA_IMAGE_W`` (2304). "
            "Use a smaller value (e.g. 200) for lighter UI.",
        ),
    ] = 2304,
    sim_vision_gt_tolerance_m: Annotated[
        float | None,
        typer.Option(
            "--sim-vision-gt-tolerance-m",
            help=(
                "With --simulate and interactive Viser: after Randomize, abort if a full "
                "synthetic scan's ingested pad/block centers differ from that layout by more "
                "than this (meters; max Euclidean over all pads/blocks). Sim-oracle check only."
            ),
        ),
    ] = None,
) -> None:
    """Run the Lab 3 Tower-of-Hanoi sequence."""
    kin = UR10e(T6t=make_T([0.0, 0.0, float(tool_z_offset)]))
    renderer = _build_renderer(
        render,
        kin,
        viser_show_live_camera=viser_live_camera,
        viser_sim_camera=viser_sim_camera,
        viser_scan_gallery_width=viser_scan_gallery_width,
    )
    use_synth = bool(
        simulate and synthetic_scan_detection and render is RenderMode.viser
    )
    use_interactive = bool(
        simulate and interactive_sim and render is RenderMode.viser
    )
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
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, float(LAB3_CAMERA_IMAGE_W))
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, float(LAB3_CAMERA_IMAGE_H))

        planned_scan: np.ndarray | None = None
        if auto_scan_points:
            planned_scan = compute_auto_scan_points(
                kin,
                T5C_DEFAULT,
                T_home=_default_home_pose(),
                verbose=controller.verbose,
            )

        task = HanoiTask(
            controller,
            detector,
            camera=camera,
            approach_height=approach_height,
            scan_settle_s=scan_settle_s,
            synthetic_scan_detection=use_synth,
            tag_grasp_orientation=tag_grasp_orientation,
            scan_points=planned_scan,
            sim_vision_gt_tolerance_m=sim_vision_gt_tolerance_m,
        )
        if use_interactive:
            from .sim import ViserRenderer

            assert isinstance(renderer, ViserRenderer)

            def _demo_worker() -> None:
                renderer.enable_interactive_sim_demo(task)
                while True:
                    renderer.wait_for_sim_demo_start()
                    task.save_demo_layout_checkpoint()
                    try:
                        if renderer._btn_randomize is not None:
                            renderer._btn_randomize.disabled = True
                        task.run(skip_sim_auto_seed=True)
                    except DemoAbort:
                        task.restore_demo_layout_checkpoint()
                        task.populate_scene()
                    except (ValueError, RuntimeError) as exc:
                        task.restore_demo_layout_checkpoint()
                        task.populate_scene()
                        if renderer._gui_demo_status is not None:
                            renderer._gui_demo_status.value = f"Scan or run failed: {exc}"
                    finally:
                        if renderer._btn_randomize is not None:
                            renderer._btn_randomize.disabled = False
                        if renderer._gui_demo_status is not None:
                            renderer._gui_demo_status.value = (
                                "Ready for another run (Start) or new layout (Randomize)."
                            )

            threading.Thread(target=_demo_worker, daemon=True).start()
            # Keep the main thread alive for Viser, but do not run ``finally`` yet:
            # ``renderer.close(wait=True)`` would block on stdin immediately and race
            # with logs / the worker. One prompt here exits the whole process after
            # the user is done with the web UI.
            try:
                input(
                    "[viser] Interactive demo is running in the browser. "
                    "Press Enter here when finished to exit...\n"
                )
            except (EOFError, KeyboardInterrupt):
                pass
        else:
            task.run()
    finally:
        if camera is not None:
            camera.release()
        controller.close()
        if renderer is not None:
            renderer.close(
                wait=(render is RenderMode.viser and not use_interactive),
            )
