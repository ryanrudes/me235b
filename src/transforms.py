"""Rigid-body pose helpers used throughout the package.

Everything is expressed as 4x4 homogeneous transforms in base (world) coordinates
unless otherwise noted. Keep this module free of robot-specific constants.

Two validation tiers are used intentionally:

- ``make_T`` is permissive for compatibility. It checks shape/finiteness and may
  clean up a rotation matrix that is already very close to SO(3), but otherwise
  stores the provided 3x3 block as-is.
- ``inv_T``, ``so3_log``, ``fuse_rigid_transforms``, and ``R_to_wxyz`` are stricter.
  They require proper rigid-body / rotation semantics, with projection only when
  the matrix is slightly off SO(3) due to numerical drift.
"""

from __future__ import annotations

import numpy as np

ArrayLike = np.ndarray

_ROT_ATOL = 1e-7
_ROT_PROJECT_TOL = 1e-3
_ROW_ATOL = 1e-8
_ROW_PROJECT_TOL = 1e-6


def _project_to_so3(R: ArrayLike) -> np.ndarray:
    """Project a 3x3 matrix to the closest proper rotation matrix in SO(3)."""
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"_project_to_so3: R must be 3x3; got {R.shape}.")
    U, _, Vt = np.linalg.svd(R)
    Rp = U @ Vt
    if np.linalg.det(Rp) < 0.0:
        U[:, -1] *= -1.0
        Rp = U @ Vt
    return Rp


def _rotation_error_metrics(R: np.ndarray) -> tuple[float, float, float]:
    """Return (orthogonality error, determinant, determinant error) for a 3x3 matrix."""
    orth_err = float(np.linalg.norm(R.T @ R - np.eye(3), ord="fro"))
    det = float(np.linalg.det(R))
    det_err = float(abs(det - 1.0))
    return orth_err, det, det_err


def _coerce_rotation_matrix(
    R: ArrayLike,
    *,
    name: str = "R",
    atol: float = _ROT_ATOL,
    project_if_close: bool = False,
    project_tol: float = _ROT_PROJECT_TOL,
) -> np.ndarray:
    """Validate a 3x3 rotation matrix, optionally projecting if it is only slightly off SO(3)."""
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"{name}: expected shape (3, 3); got {R.shape}.")
    if not np.all(np.isfinite(R)):
        raise ValueError(f"{name}: matrix contains non-finite values.")

    orth_err, det, det_err = _rotation_error_metrics(R)
    if det > 0.0 and orth_err <= atol and det_err <= atol:
        return R.copy()

    if project_if_close and det > 0.0 and orth_err <= project_tol and det_err <= project_tol:
        return _project_to_so3(R)

    raise ValueError(
        f"{name}: matrix is not a valid rotation "
        f"(orth_err={orth_err:.3e}, det={det:.6f})."
    )


def _coerce_rigid_transform(
    T: ArrayLike,
    *,
    name: str = "T",
    row_atol: float = _ROW_ATOL,
    row_project_tol: float = _ROW_PROJECT_TOL,
    project_rotation_if_close: bool = False,
    rot_project_tol: float = _ROT_PROJECT_TOL,
) -> np.ndarray:
    """Validate a 4x4 rigid transform with bottom row [0, 0, 0, 1].

    Slight bottom-row drift is repaired when it is clearly just numerical noise.
    Clearly invalid homogeneous rows still raise ``ValueError``.
    """
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"{name}: expected shape (4, 4); got {T.shape}.")
    if not np.all(np.isfinite(T)):
        raise ValueError(f"{name}: transform contains non-finite values.")

    row_target = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    Tc = T.copy()

    if np.allclose(Tc[3, :], row_target, atol=row_atol):
        Tc[3, :] = row_target
    elif np.allclose(Tc[3, :], row_target, atol=row_project_tol):
        Tc[3, :] = row_target
    else:
        raise ValueError(
            f"{name}: bottom row must be [0, 0, 0, 1] within atol={row_project_tol}."
        )

    Tc[:3, :3] = _coerce_rotation_matrix(
        Tc[:3, :3],
        name=f"{name}[:3, :3]",
        project_if_close=project_rotation_if_close,
        project_tol=rot_project_tol,
    )
    return Tc


def make_T(pos, R=None) -> np.ndarray:
    """Build a 4x4 homogeneous transform from a 3-vector and a 3x3 rotation matrix.

    This constructor is intentionally permissive for compatibility:
    - it validates shape and finiteness
    - it projects ``R`` onto SO(3) only when ``R`` is already very close
    - otherwise it preserves the provided 3x3 block as-is

    Use stricter helpers such as ``inv_T``, ``so3_log``, or ``R_to_wxyz`` when
    rigid-body semantics must be enforced.
    """
    T = np.eye(4, dtype=float)
    T[:3, 3] = np.asarray(pos, dtype=float).reshape(3)
    if R is not None:
        R = np.asarray(R, dtype=float).reshape(3, 3)
        if not np.all(np.isfinite(R)):
            raise ValueError("make_T: rotation block contains non-finite values.")

        orth_err, det, det_err = _rotation_error_metrics(R)
        if (
            det > 0.0
            and orth_err > _ROT_ATOL
            and orth_err <= _ROT_PROJECT_TOL
            and det_err <= _ROT_PROJECT_TOL
        ):
            R = _project_to_so3(R)

        T[:3, :3] = R
    return T


def make_T_rpy(pos, rpy) -> np.ndarray:
    """Build a 4x4 homogeneous transform from a position and (roll, pitch, yaw) in radians.

    Rotation convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    r, p, y = (float(v) for v in np.asarray(rpy, dtype=float).reshape(3))
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]], dtype=float)
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]], dtype=float)
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]], dtype=float)
    return make_T(pos, Rz @ Ry @ Rx)


def inv_T(T: ArrayLike) -> np.ndarray:
    """Inverse of a rigid-body 4x4 transform, exploiting R^T for the rotation block.

    Slight numerical drift in the rotation block or homogeneous bottom row is
    repaired first; clearly invalid rigid transforms raise ``ValueError``.
    """
    T = _coerce_rigid_transform(T, name="T", project_rotation_if_close=True)
    R = T[:3, :3]
    p = T[:3, 3]
    Ti = np.eye(4, dtype=float)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ p
    return Ti


def wrap_to_pi(x):
    """Wrap angles to [-pi, pi).

    Note: values exactly equal to ``pi`` map to ``-pi`` under this convention.
    Works element-wise on arrays.
    """
    x = np.asarray(x, dtype=float)
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def shortest_joint_motion_target(
    q_from: np.ndarray, q_to: np.ndarray,
) -> np.ndarray:
    """Same configuration as ``q_to``, continuing smoothly from ``q_from``.

    IK returns angles in ``[-π, π)``; the next solve may pick an equivalent
    branch (often wrist joints) differing by ``2π``. Linear joint interpolation
    would then sweep a full turn; here each joint changes by at most ``π``.
    """
    q_from = np.asarray(q_from, dtype=float).reshape(6)
    q_to = np.asarray(q_to, dtype=float).reshape(6)
    return q_from + wrap_to_pi(q_to - q_from)


def skew_symmetric(v: ArrayLike) -> np.ndarray:
    """3×3 skew matrix ``[v]_×`` so that ``[v]_× @ w == v × w``."""
    v = np.asarray(v, dtype=float).reshape(3)
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=float,
    )


def so3_exp(omega: ArrayLike) -> np.ndarray:
    """SO(3) exponential map: rotation vector ``omega`` (axis × angle) → 3×3 ``R``."""
    omega = np.asarray(omega, dtype=float).reshape(3)
    th = float(np.linalg.norm(omega))
    if th < 1e-12:
        return np.eye(3, dtype=float)
    axis = omega / th
    K = skew_symmetric(axis)
    s, c = float(np.sin(th)), float(np.cos(th))
    return np.eye(3, dtype=float) + s * K + (1.0 - c) * (K @ K)


def so3_log(R: ArrayLike) -> np.ndarray:
    """SO(3) logarithm. Returns a rotation vector (axis * angle) for a 3x3 matrix.

    Slightly non-orthonormal inputs are projected onto SO(3); clearly invalid
    inputs raise ``ValueError``.
    """
    R = _coerce_rotation_matrix(R, name="R", project_if_close=True)

    tr = float(np.trace(R))
    c = float(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))
    th = float(np.arccos(c))

    if th < 1e-12:
        return np.zeros(3, dtype=float)

    s = float(np.sin(th))

    if abs(s) < 1e-12:
        axis = np.sqrt(np.maximum(0.0, (np.diag(R) + 1.0) / 2.0)).astype(float)
        if (R[2, 1] - R[1, 2]) < 0:
            axis[0] = -axis[0]
        if (R[0, 2] - R[2, 0]) < 0:
            axis[1] = -axis[1]
        if (R[1, 0] - R[0, 1]) < 0:
            axis[2] = -axis[2]

        n = float(np.linalg.norm(axis))
        if n < 1e-12:
            raise ValueError("so3_log: could not determine axis for a near-pi rotation.")
        axis = axis / n
        return th * axis

    axis = (1.0 / (2.0 * s)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=float,
    )
    return th * axis


def tool_orientation_tilted(yaw_deg: float = 35.0, roll_deg: float = 10.0) -> np.ndarray:
    """Rotation matrix for a roughly-downward tool frame with a small tilt.

    Using a pure axis-aligned "down" orientation is near-singular for the UR10e IK,
    so we apply a small yaw + roll to keep the wrist out of the singularity.
    """
    ry = float(np.pi)
    y = float(np.deg2rad(yaw_deg))
    r = float(np.deg2rad(roll_deg))

    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]], dtype=float)
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]], dtype=float)
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]], dtype=float)

    return Rz @ Ry @ Rx


def _robust_inlier_mask_positive(
    values: np.ndarray,
    *,
    mad_k: float,
) -> np.ndarray:
    """Inliers for nonnegative scores (e.g. distances): keep ``<= med + mad_k * σ``."""
    v = np.asarray(values, dtype=float).reshape(-1)
    if v.size == 0:
        return np.zeros(0, dtype=bool)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    if mad < 1e-12:
        return np.ones(v.shape[0], dtype=bool)
    thresh = med + mad_k * 1.4826 * mad
    return v <= thresh


def fuse_rigid_transforms(
    transforms: list[np.ndarray],
    *,
    trans_mad_k: float = 3.5,
    rot_mad_k: float = 3.5,
) -> np.ndarray:
    """Pool several ``4×4`` rigid estimates of the *same* body pose into one ``T``.

    **Translation**: robust center ``t_med = median(t_i)``, Euclidean distances
    ``d_i = ‖t_i - t_med‖``, MAD-based rejection, then ``median`` of surviving
    positions.

    **Rotation**: among translation inliers, pick ``R_ref`` from the sample whose
    translation is closest to ``t_med``. Map each ``R_i`` to
    ``ω_i = log(R_ref^T R_i)`` in ``so(3)``, reject outliers on ``‖ω_i‖`` with the
    same MAD rule, then ``R = R_ref @ exp(mean(ω))``.

    Slightly non-orthonormal rotation blocks are projected onto SO(3) before fusion.
    Falls back to “keep all samples” when MAD degenerates (e.g. a single cluster
    or only one measurement).
    """
    if not transforms:
        raise ValueError("fuse_rigid_transforms: empty list.")
    Ts = [
        _coerce_rigid_transform(
            T,
            name=f"transforms[{i}]",
            project_rotation_if_close=True,
        )
        for i, T in enumerate(transforms)
    ]
    n = len(Ts)
    if n == 1:
        return Ts[0].copy()

    pos = np.stack([T[:3, 3] for T in Ts], axis=0)
    t_med = np.median(pos, axis=0)
    d_trans = np.linalg.norm(pos - t_med, axis=1)
    m_t = _robust_inlier_mask_positive(d_trans, mad_k=trans_mad_k)
    if not np.any(m_t):
        m_t = np.ones(n, dtype=bool)

    Ts_t = [Ts[i] for i in range(n) if m_t[i]]
    pos_t = pos[m_t]
    t_anchor = np.median(pos_t, axis=0)
    j = int(np.argmin(np.linalg.norm(pos_t - t_anchor, axis=1)))
    R_ref = Ts_t[j][:3, :3].copy()

    omegas = np.stack(
        [so3_log(R_ref.T @ T[:3, :3]) for T in Ts_t],
        axis=0,
    )
    d_rot = np.linalg.norm(omegas, axis=1)
    m_r = _robust_inlier_mask_positive(d_rot, mad_k=rot_mad_k)
    if not np.any(m_r):
        m_r = np.ones(len(Ts_t), dtype=bool)

    Ts_tr = [Ts_t[i] for i in range(len(Ts_t)) if m_r[i]]
    om_tr = omegas[m_r]
    omega_mean = np.mean(om_tr, axis=0)
    R_fused = _project_to_so3(R_ref @ so3_exp(omega_mean))

    pos_tr = np.stack([T[:3, 3] for T in Ts_tr], axis=0)
    t_fused = np.median(pos_tr, axis=0)

    return make_T(t_fused, R_fused)


def as_6vec(x) -> np.ndarray:
    """Coerce a 6-element array-like into a flat float64 np.ndarray of shape (6,)."""
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size != 6:
        raise ValueError(f"Expected a 6-vector; got shape {np.asarray(x).shape} with {arr.size} elements.")
    return arr


def R_to_wxyz(R: ArrayLike) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a unit quaternion in (w, x, y, z) order.

    Uses Shepperd's method for numerical stability near singular cases.
    Clean rotations take the fast path; slightly noisy rotations are projected
    onto SO(3); clearly invalid inputs raise ``ValueError``.
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"R_to_wxyz: R must be 3x3; got {R.shape}.")
    if not np.all(np.isfinite(R)):
        raise ValueError("R_to_wxyz: matrix contains non-finite values.")

    orth_err, det, det_err = _rotation_error_metrics(R)
    if not (det > 0.0 and orth_err <= _ROT_ATOL and det_err <= _ROT_ATOL):
        R = _coerce_rotation_matrix(R, name="R", project_if_close=True)

    tr = float(np.trace(R))
    if tr > 0.0:
        s = float(np.sqrt(tr + 1.0) * 2.0)
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = float(np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0)
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = float(np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0)
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = float(np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0)
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=float)
    qn = float(np.linalg.norm(q))
    if qn < 1e-12:
        raise ValueError("R_to_wxyz: quaternion norm underflow.")
    return q / qn
