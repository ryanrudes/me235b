"""Numerical IK for the link-5 camera mount + Lab-3 intrinsics bundle sizing."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from me235b.kinematics import UR10e
from me235b.transforms import make_T_rpy
from me235b.sim import (
    LAB3_CAMERA_IMAGE_H,
    LAB3_CAMERA_IMAGE_W,
    LAB3_CAMERA_K,
    T5C_LAB3,
    _K_for_vertically_flipped_image,
    _lab3_scaled_pinhole_bundle,
    _simple_pinhole_bundle,
)


def test_lab3_synthetic_resolution_matches_hardware() -> None:
    wh, _, _ = _lab3_scaled_pinhole_bundle(
        (LAB3_CAMERA_IMAGE_W, LAB3_CAMERA_IMAGE_H), use_distortion=True
    )
    assert wh == (LAB3_CAMERA_IMAGE_W, LAB3_CAMERA_IMAGE_H)


def test_lab3_bundle_honors_width_and_height() -> None:
    wh, K, D = _lab3_scaled_pinhole_bundle((320, 240), use_distortion=False)
    assert wh == (320, 240)
    assert K.shape == (3, 3)
    assert np.allclose(D, 0.0)


def test_lab3_bundle_derives_height_from_width() -> None:
    native_w = 2.0 * float(LAB3_CAMERA_K[0, 2])
    native_h = 2.0 * float(LAB3_CAMERA_K[1, 2])
    wh, _, _ = _lab3_scaled_pinhole_bundle((400, 0), use_distortion=False)
    W, H = wh
    assert W == 400
    assert H == int(round(400 * native_h / native_w))


def test_simple_bundle_matches_vga_and_zero_distortion() -> None:
    wh, K, D = _simple_pinhole_bundle((LAB3_CAMERA_IMAGE_W, LAB3_CAMERA_IMAGE_H))
    assert wh == (LAB3_CAMERA_IMAGE_W, LAB3_CAMERA_IMAGE_H)
    assert np.allclose(D, 0.0)
    assert K.shape == (3, 3)
    W, H = wh
    hfov = 58.0 * np.pi / 180.0
    fx_expect = (0.5 * W) / np.tan(0.5 * hfov)
    assert abs(float(K[0, 0]) - fx_expect) < 1e-6
    assert abs(float(K[1, 1]) - fx_expect) < 1e-6
    assert abs(float(K[0, 2]) - 0.5 * W) < 1e-9
    assert abs(float(K[1, 2]) - 0.5 * H) < 1e-9


def test_simple_bundle_derives_height_from_width_4_3() -> None:
    wh, _, _ = _simple_pinhole_bundle((400, 0))
    W, H = wh
    assert W == 400
    assert H == int(round(400 * 3.0 / 4.0))


def test_K_vertical_flip_matches_row_flip_identity() -> None:
    """v' = H-1-v  <=>  K' with neg fy and shifted cy (pinhole)."""
    _, K, _ = _simple_pinhole_bundle((640, 480))
    H = 480
    Kf = _K_for_vertically_flipped_image(K, H)
    assert Kf[0, 0] == K[0, 0] and Kf[0, 2] == K[0, 2]
    assert Kf[1, 1] == -K[1, 1]
    assert Kf[1, 2] == float(H - 1) - K[1, 2]
    # one point: flipped image row should match Kf mapping
    X, Y, Z = 0.01, -0.02, 0.5
    r = np.zeros((3, 1))
    t = np.zeros((3, 1))
    pts, _ = cv2.projectPoints(np.array([[X, Y, Z]], np.float64), r, t, K, None)
    u, v = float(pts[0, 0, 0]), float(pts[0, 0, 1])
    pts2, _ = cv2.projectPoints(np.array([[X, Y, Z]], np.float64), r, t, Kf, None)
    u2, v2 = float(pts2[0, 0, 0]), float(pts2[0, 0, 1])
    assert abs(u2 - u) < 1e-5
    assert abs(v2 - (H - 1 - v)) < 1e-5


def test_ik_camera_mount_roundtrip() -> None:
    kin = UR10e()
    T5c = np.asarray(T5C_LAB3, dtype=float)
    q_true = np.array([0.15, -0.65, 0.55, -0.95, -0.45, 0.25], dtype=float)
    assert kin.safety_check(q_true)

    T_tgt = kin.fk_camera_on_link5(q_true, T5c)
    theta_seed = kin.classical_to_dh_modified(q_true)
    th, _, info = kin.ik_camera_mount(T_tgt, T5c, theta_seed=theta_seed)
    assert info.get("success"), info.get("message")
    q_hat = kin.dh_modified_to_classical(th)
    T_hat = kin.fk_camera_on_link5(q_hat, T5c)

    pos_err = float(np.linalg.norm(T_tgt[:3, 3] - T_hat[:3, 3]))
    R_err = T_hat[:3, :3].T @ T_tgt[:3, :3]
    tr = float(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0))
    ang = float(np.arccos(tr))
    assert pos_err < 2e-3
    assert ang < 2e-2


def test_ik_camera_mount_cold_start() -> None:
    kin = UR10e()
    T5c = np.asarray(T5C_LAB3, dtype=float)
    q_safe = np.array([0.1, -0.85, 1.05, -1.55, -0.95, 0.15], dtype=float)
    assert kin.safety_check(q_safe)
    T_tgt = kin.fk_camera_on_link5(q_safe, T5c)
    th, _, info = kin.ik_camera_mount(T_tgt, T5c, theta_seed=np.zeros(6))
    assert info.get("success"), info.get("message")
    q_hat = kin.dh_modified_to_classical(th)
    T_hat = kin.fk_camera_on_link5(q_hat, T5c)
    assert float(np.linalg.norm(T_tgt[:3, 3] - T_hat[:3, 3])) < 5e-3


def test_ik_camera_mount_scan_grid_weighted_pose() -> None:
    """Lab3 scan: weighted IK hits grid position while biasing toward downward RPY."""
    pytest.importorskip("scipy")
    kin = UR10e()
    T5c = np.asarray(T5C_LAB3, dtype=float)
    q_home = np.array([-0.5715, -1.2007, -2.0028, -1.5089, 1.5708, -2.1423], dtype=float)
    assert kin.safety_check(q_home)
    seed = kin.classical_to_dh_modified(q_home)
    T_tgt = make_T_rpy([-0.35, -0.23, 0.5], [np.pi, 0.0, 0.0])
    th, _, info = kin.ik_camera_mount(
        T_tgt,
        T5c,
        theta_seed=seed,
        position_only=False,
        trans_weight=20.0,
        rot_weight=1.0,
        pos_tol=1.0e-2,
        rot_tol=0.85,
        max_iter=120,
        pre_solve_position=True,
    )
    assert info.get("success"), info.get("message")
    assert info.get("message") in (
        "ik_camera_mount: scipy weighted refine.",
        "ik_camera_mount: position-only fallback (scipy refine missed pose tolerances).",
    )
    assert float(info.get("pos_err_m", 1.0)) <= 1.0e-2
    assert float(info.get("rot_err_rad", 10.0)) <= 2.0
    q_hat = kin.dh_modified_to_classical(th)
    assert kin.safety_check(q_hat)
