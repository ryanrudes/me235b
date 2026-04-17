"""Rigid-body pose helpers used throughout the package.

Everything is expressed as 4x4 homogeneous transforms in base (world) coordinates
unless otherwise noted. Keep this module free of robot-specific constants.
"""

from __future__ import annotations

import numpy as np

ArrayLike = np.ndarray


def make_T(pos, R=None) -> np.ndarray:
    """Build a 4x4 homogeneous transform from a 3-vector and a 3x3 rotation matrix."""
    T = np.eye(4, dtype=float)
    T[:3, 3] = np.asarray(pos, dtype=float).reshape(3)
    if R is not None:
        R = np.asarray(R, dtype=float).reshape(3, 3)
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
    """Inverse of a rigid-body 4x4 transform, exploiting R^T for the rotation block."""
    T = np.asarray(T, dtype=float)
    R = T[:3, :3]
    p = T[:3, 3]
    Ti = np.eye(4, dtype=float)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ p
    return Ti


def wrap_to_pi(x):
    """Wrap angles to (-pi, pi]. Works element-wise on arrays."""
    x = np.asarray(x, dtype=float)
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def so3_log(R: ArrayLike) -> np.ndarray:
    """SO(3) logarithm. Returns a rotation vector (axis * angle) for a 3x3 matrix."""
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"so3_log: R must be 3x3; got {R.shape}.")

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


def as_6vec(x) -> np.ndarray:
    """Coerce a 6-element array-like into a flat float64 np.ndarray of shape (6,)."""
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size != 6:
        raise ValueError(f"Expected a 6-vector; got shape {np.asarray(x).shape} with {arr.size} elements.")
    return arr


def R_to_wxyz(R: ArrayLike) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a unit quaternion in (w, x, y, z) order.

    Uses Shepperd's method for numerical stability near singular cases.
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"R_to_wxyz: R must be 3x3; got {R.shape}.")

    tr = float(np.trace(R))
    if tr > 0:
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
    return np.array([qw, qx, qy, qz], dtype=float)
