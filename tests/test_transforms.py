"""Tests for SE(3) helpers used in scan fusion."""

from __future__ import annotations

import numpy as np

from me235b.transforms import fuse_rigid_transforms, make_T, so3_exp, so3_log


def test_so3_exp_log_roundtrip_small() -> None:
    omega = np.array([0.12, -0.05, 0.03], dtype=float)
    R = so3_exp(omega)
    w2 = so3_log(R)
    R2 = so3_exp(w2)
    assert np.linalg.norm(R - R2) < 1e-9
    assert np.linalg.norm(omega - w2) < 1e-9


def test_fuse_rigid_transforms_rejects_translation_outlier() -> None:
    R = np.eye(3)
    cluster = [
        make_T([0.0, 0.0, 0.0], R),
        make_T([0.001, 0.0, 0.0], R),
        make_T([0.002, 0.0, 0.0], R),
        make_T([0.2, 0.0, 0.0], R),
    ]
    T = fuse_rigid_transforms(cluster, trans_mad_k=3.5, rot_mad_k=3.5)
    # Inliers cluster near 1 mm; 20 cm sample should be dropped.
    assert abs(float(T[0, 3]) - 0.001) < 2e-4
    assert abs(float(T[1, 3])) < 2e-4
    assert abs(float(T[2, 3])) < 2e-4


def test_fuse_rigid_transforms_rejects_rotation_outlier() -> None:
    t = np.array([0.01, 0.02, 0.0], dtype=float)
    R0 = np.eye(3)
    # Small perturbations (inliers)
    w_small = np.deg2rad(np.array([1.0, -0.5, 0.3]))
    R1 = R0 @ so3_exp(w_small)
    R2 = R0 @ so3_exp(1.2 * w_small)
    # Large spurious yaw (outlier)
    R_bad = R0 @ so3_exp(np.deg2rad([0.0, 0.0, 45.0]))
    Ts = [
        make_T(t, R1),
        make_T(t + np.array([0.0005, 0.0, 0.0]), R2),
        make_T(t, R_bad),
    ]
    T = fuse_rigid_transforms(Ts, trans_mad_k=5.0, rot_mad_k=3.0)
    Rf = T[:3, :3]
    # Fused rotation should stay near inlier cluster, not 45° off
    w_rel = so3_log(R0.T @ Rf)
    assert np.linalg.norm(w_rel) < np.deg2rad(12.0)


def test_fuse_rigid_transforms_single_is_identity_copy() -> None:
    T0 = make_T([1.0, 2.0, 3.0], np.eye(3))
    T1 = fuse_rigid_transforms([T0])
    assert np.allclose(T0, T1)
