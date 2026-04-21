"""Offline scan path planning (link-5 camera IK over the tag workspace)."""

from __future__ import annotations

import numpy as np
import pytest

from me235b.hanoi import (
    T5C_DEFAULT,
    _chain_validate_scan_points,
    _default_home_pose,
    compute_auto_scan_points,
)
from me235b.kinematics import UR10e
from me235b.transforms import make_T


@pytest.mark.parametrize("tool_z", (0.0, 0.20))
def test_compute_auto_scan_points_feasible(tool_z: float) -> None:
    pytest.importorskip("scipy")
    kin = UR10e(T6t=np.eye(4) if tool_z == 0.0 else make_T([0.0, 0.0, float(tool_z)]))
    T5c = np.asarray(T5C_DEFAULT, dtype=float)
    T_home = _default_home_pose()
    pts = compute_auto_scan_points(kin, T5c, T_home=T_home, verbose=False)
    assert pts.ndim == 2 and pts.shape[1] == 3
    assert pts.shape[0] >= 4

    again = _chain_validate_scan_points(kin, T5c, [pts[i] for i in range(pts.shape[0])], T_home=T_home)
    assert len(again) == pts.shape[0]


def test_compute_auto_scan_points_max_points_spread() -> None:
    pytest.importorskip("scipy")
    kin = UR10e(T6t=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.20], [0, 0, 0, 1]], dtype=float))
    T5c = np.asarray(T5C_DEFAULT, dtype=float)
    full = compute_auto_scan_points(kin, T5c, T_home=_default_home_pose())
    capped = compute_auto_scan_points(kin, T5c, T_home=_default_home_pose(), max_points=6)
    assert capped.shape[0] <= 6
    assert capped.shape[0] >= 1
    if full.shape[0] > 6:
        assert capped.shape[0] == 6
