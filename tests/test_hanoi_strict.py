"""Fast checks for strict-vision / world-model gating (no Viser)."""

from __future__ import annotations

import numpy as np
import pytest

from me235b.hanoi import BoxTag, HanoiDetector, HanoiTask, PadTag
from me235b.kinematics import UR10e
from me235b.robot import RobotController
from me235b.sim import NullRenderer


def test_require_full_world_model_raises_when_tags_missing() -> None:
    """After scan, empty world model must abort (hardware and strict sim)."""
    kin = UR10e()
    ctrl = RobotController(kin, simulate=True, renderer=NullRenderer())
    det = HanoiDetector()
    task = HanoiTask(
        ctrl,
        det,
        camera=None,
        synthetic_scan_detection=True,
    )
    with pytest.raises(ValueError, match="did not find all required tags"):
        task._require_full_world_model()


def test_sim_vision_gt_tolerance_raises_on_drift() -> None:
    kin = UR10e()
    ctrl = RobotController(kin, simulate=True, renderer=NullRenderer())
    det = HanoiDetector()
    task = HanoiTask(
        ctrl,
        det,
        camera=None,
        synthetic_scan_detection=True,
        sim_vision_gt_tolerance_m=0.001,
    )
    z0 = np.zeros(3, dtype=float)
    task.pad_centers = {p: z0.copy() for p in PadTag}
    task.block_centers = {b: z0.copy() for b in BoxTag}
    task._sim_gt_pad_centers = {p: z0.copy() for p in PadTag}
    task._sim_gt_block_centers = {b: z0.copy() for b in BoxTag}
    task.pad_centers[PadTag.START_PAD] = np.array([0.01, 0.0, 0.0], dtype=float)
    with pytest.raises(ValueError, match="Sim vision vs Randomize layout"):
        task._raise_if_sim_vision_gt_exceeds_tolerance()


def test_sim_vision_gt_tolerance_passes_when_within_tol() -> None:
    kin = UR10e()
    ctrl = RobotController(kin, simulate=True, renderer=NullRenderer())
    det = HanoiDetector()
    task = HanoiTask(
        ctrl,
        det,
        camera=None,
        synthetic_scan_detection=True,
        sim_vision_gt_tolerance_m=0.01,
    )
    z0 = np.zeros(3, dtype=float)
    task.pad_centers = {p: z0.copy() for p in PadTag}
    task.block_centers = {b: z0.copy() for b in BoxTag}
    task._sim_gt_pad_centers = {p: z0.copy() for p in PadTag}
    task._sim_gt_block_centers = {b: z0.copy() for b in BoxTag}
    task.pad_centers[PadTag.START_PAD] = np.array([0.005, 0.0, 0.0], dtype=float)
    task._raise_if_sim_vision_gt_exceeds_tolerance()
