"""Lab Part 1: print ``T_base_gripper`` from a test image + live robot joints.

Reads classical joint angles from the UR (:func:`q_class_from_robot`), builds
``T_base_cam = fk_to_frame(q,5) @ T5c``, runs ArUco with
:class:`me235b.detector.ArucoDetector` intrinsics.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import typer
from typing_extensions import Annotated

from .detector import ArucoDetector
from .hanoi import GRASP_ORIENTATION, T5C_DEFAULT
from .kinematics import UR10e

BLOCK_WIDTH = 0.08
BLOCK_THICKNESS = 0.05


def T_cam_marker(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rvec = np.asarray(rvec, dtype=float).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=float).reshape(3)
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = tvec
    return T


def T_marker_block_center(marker_length: float) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[0, 3] = BLOCK_WIDTH / 2.0
    T[1, 3] = marker_length / 2.0
    T[2, 3] = -BLOCK_THICKNESS / 2.0
    return T


def T_base_cam_from_joints(q_class: np.ndarray) -> np.ndarray:
    """``fk_to_frame(q, 5) @ T5c`` (camera in base for that arm pose)."""
    q = np.asarray(q_class, dtype=float).reshape(6)
    return UR10e().fk_to_frame(q, 5) @ T5C_DEFAULT


def q_class_from_robot(ip: str) -> np.ndarray:
    """Current joint vector (radians), same convention as :meth:`RobotController.movej`."""
    try:
        import urx  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Install package 'urx' to read joints from the robot (e.g. uv add urx)."
        ) from exc
    robot = urx.Robot(ip)
    try:
        q = np.asarray(robot.getj(), dtype=float).reshape(6)
    finally:
        robot.close()
    return q


def grasp_T_base_gripper(
    T_base_cam: np.ndarray,
    T_cam_marker: np.ndarray,
    *,
    marker_length: float,
) -> np.ndarray:
    """4×4 base ← open gripper center (fixed lab grasp orientation)."""
    T_base_cam = np.asarray(T_base_cam, dtype=float).reshape(4, 4)
    T_cam_marker = np.asarray(T_cam_marker, dtype=float).reshape(4, 4)
    T_bm = T_base_cam @ T_cam_marker
    T_bb = T_bm @ T_marker_block_center(marker_length)

    Tg = np.eye(4, dtype=float)
    Tg[:3, :3] = np.asarray(GRASP_ORIENTATION, dtype=float).copy()
    Tg[:3, 3] = T_bb[:3, 3]
    return Tg


def grasp_transform(
    image: Annotated[Path, typer.Argument(help="Test image (BGR) with the block tag.")],
    tag_id: Annotated[int, typer.Argument(help="ArUco id on that block (e.g. 3, 4, 5).")],
    robot_ip: Annotated[
        str,
        typer.Argument(help="UR controller host or IP (arm must match practice camera pose)."),
    ],
) -> None:
    """Print ``T_base_gripper`` (4×4). Joints are read live — park the arm first."""
    img = cv2.imread(str(image))
    if img is None:
        raise typer.BadParameter(f"Could not read image: {image}")

    detector = ArucoDetector()
    T_bc = T_base_cam_from_joints(q_class_from_robot(robot_ip))

    found = None
    for mid, rvec, tvec in detector.find_tags(img):
        if int(mid) == int(tag_id):
            found = (rvec, tvec)
            break
    if found is None:
        raise typer.BadParameter(f"No tag id={tag_id} in image.")

    T = grasp_T_base_gripper(
        T_bc, T_cam_marker(found[0], found[1]), marker_length=detector.marker_length
    )
    np.set_printoptions(precision=6, suppress=True)
    print(T)


if __name__ == "__main__":
    typer.run(grasp_transform)
