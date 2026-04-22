"""Tests for me235b.grasp_from_tag."""

import numpy as np

from me235b.grasp_from_tag import (
    BLOCK_THICKNESS_M,
    BLOCK_WIDTH_M,
    T_base_cam_from_joints,
    grasp_T_base_gripper,
)
from me235b.hanoi import GRASP_ORIENTATION, T5C_DEFAULT
from me235b.kinematics import UR10e


def test_block_constants_match_lab_defaults() -> None:
    assert BLOCK_WIDTH_M == 0.08
    assert BLOCK_THICKNESS_M == 0.05


def test_grasp_T_base_gripper_identity_chain() -> None:
    T_bc = np.eye(4)
    T_cm = np.eye(4)
    T_cm[:3, 3] = [1.0, 2.0, 3.0]
    Tg = grasp_T_base_gripper(T_bc, T_cm, marker_length_m=0.02)
    np.testing.assert_allclose(Tg[:3, 3], [1.04, 2.01, 2.975])
    np.testing.assert_allclose(Tg[:3, :3], GRASP_ORIENTATION)


def test_T_base_cam_from_joints_matches_fk_chain() -> None:
    kin = UR10e()
    q = np.array([0.1, -0.2, 0.3, -0.15, 0.05, -0.1], dtype=float)
    expect = kin.fk_to_frame(q, 5) @ T5C_DEFAULT
    np.testing.assert_allclose(T_base_cam_from_joints(q), expect)
