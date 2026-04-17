import numpy as np
import pickle
import matplotlib.pyplot as plt
import urx
import time



# wrappers for robot
def movej_fn(q_class, **kwargs):
    robot.movej(tuple(np.asarray(q_class).tolist()), aj, vj, wait=True)
    robot.stopj(aj)

def translate_fn(robot_unused, dx, dy, dz):
    robot.translate((dx, dy, dz), al, vl)
    robot.stopl(al)



def dh_classical(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """
    Classical DH transform:
      T = Rz(theta) * Tz(d) * Tx(a) * Rx(alpha)
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,     sa,       ca,      d],
        [0.0,    0.0,      0.0,    1.0],
    ], dtype=float)


def fk(q, T6t=None) -> np.ndarray:
    """
    FK using classical DH for UR10e.

    Parameters
    ----------
    q : array-like (6,)
        Joint angles [rad].
    T6t : (4,4) array-like, optional
        Frame6 to tool transform. If None: identity.

    Returns
    -------
    T_Bt : (4,4) ndarray
        Base to tool transform.
    """
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.size != 6:
        raise ValueError(f"fk expects 6 joint angles; got {q.shape}.")

    if T6t is None:
        T6t = np.eye(4, dtype=float)
    else:
        T6t = np.asarray(T6t, dtype=float)
        if T6t.shape != (4, 4):
            raise ValueError(f"T6t must be 4x4; got {T6t.shape}.")

    a = np.array([0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0], dtype=float)
    d = np.array([0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655], dtype=float)
    alpha = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0], dtype=float)

    T_B6 = np.eye(4, dtype=float)
    for i in range(6):
        T_B6 = T_B6 @ dh_classical(a[i], alpha[i], d[i], q[i])

    return T_B6 @ T6t

def read_pickle(filename: str):
    """
    reads in pickle file
    """
    with open(filename, "rb") as f:
        return pickle.load(f)


def _as_1d_float_array(x, expected_len=6) -> np.ndarray:
    """
    Helper: transforms a joint vector into shape (6,).
    """
    arr = np.asarray(x, dtype=float).reshape(-1)
    if expected_len is not None and arr.size != expected_len:
        raise ValueError(f"Expected vector of length {expected_len}, got shape {np.asarray(x).shape} with {arr.size} elems.")
    return arr


def safety_check(theta, T6t=None) -> bool:
    """
    Checks whether any frame origin (after each DH step) is below z=0,
    and additionally checks the tool point (frame6 * T6t).

    Parameters
    ----------
    theta : array-like, shape (6,)
        Joint angles [rad]
    T6t : array-like, shape (4,4), optional
        Frame6 to tool transform. Defaults to identity.

    Returns
    -------
    is_safe : bool
    """
    theta = _as_1d_float_array(theta, expected_len=6)

    if T6t is None:
        T6t = np.eye(4, dtype=float)
    else:
        T6t = np.asarray(T6t, dtype=float)
        if T6t.shape != (4, 4):
            raise ValueError(f"T6t must be 4x4; got {T6t.shape}.")

    a = np.array([0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0], dtype=float)
    d = np.array([0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655], dtype=float)
    alpha = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0], dtype=float)

    from math import isfinite  

    T = np.eye(4, dtype=float)

    for i in range(6):
        T = T @ dh_classical(a[i], alpha[i], d[i], theta[i])
        origin = T[0:3, 3]
        if origin[2] < 0.0:
            return False

    T_tool = T @ T6t
    if T_tool[2, 3] < 0.0:
        return False

    return True


def _normalize_segments_structure(data):
    """
    helper: generator that yields (theta_start, theta_end) pairs.
    """
    for letter_idx, segments in enumerate(data):
        for seg_idx, seg in enumerate(segments):
            if isinstance(seg, (list, tuple)) and len(seg) == 2:
                yield seg[0], seg[1]
            else:
                raise ValueError(
                    f"Unrecognized segment format at letter {letter_idx}, segment {seg_idx}: {type(seg)} / {seg}"
                )


def plot_or_execute_word(
    filename: str,
    fk_func,
    T6t=None,
    ax=None,
    show=True,
    *,
    execute=False,              # if True, command robot instead of plotting
    robot=None,                 # urx.Robot instance (required if execute=True and dry_run=False)
    dry_run=False,              # if True, don't command; just print what would be sent
    movej_fn=None,
    movej_kwargs=None,          # optional kwargs for movej_fn or robot.movej fallback
    external_safety_filter=None,# TA safety filter on classical joints
    dedup_consecutive=True,     # skip repeated end joints
    verbose=True,
):
    """
    Plot OR execute a word from a pickle file that stores joint-angle segments.

    Your pickle is assumed to decode via:
        for theta_start, theta_end in _normalize_segments_structure(data): ...

    Modes
    -----
    execute=False (default):
        same as your original plot_word: plots XY projection using FK.

    execute=True:
        commands the robot to traverse each segment using joint motions:
            movej(theta_start); movej(theta_end)

        This is the most direct way to execute segment data stored in joint space.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if movej_kwargs is None:
        movej_kwargs = {}

    # ---- tool offset ----
    if T6t is None:
        T6t = np.eye(4, dtype=float)
        T6t[2, 3] = 0.30  # 30cm along z (your default)
    else:
        T6t = np.asarray(T6t, dtype=float)
        if T6t.shape != (4, 4):
            raise ValueError(f"T6t must be 4x4; got {T6t.shape}.")

    data = read_pickle(filename)
    segments = list(_normalize_segments_structure(data))

    if not execute:
        # ---------- PLOT MODE ----------
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_aspect("equal", adjustable="box")

        for theta_start, theta_end in segments:
            theta_start = _as_1d_float_array(theta_start, expected_len=6)
            theta_end   = _as_1d_float_array(theta_end, expected_len=6)

            T1 = fk_func(theta_start, T6t)
            T2 = fk_func(theta_end, T6t)

            p1 = T1[0:2, 3]
            p2 = T2[0:2, 3]

            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k")

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("Projected Word in XY Plane")

        if show:
            plt.show()
        return ax

    # ---------- EXECUTE MODE ----------
    # In execute mode we do not require matplotlib
    if dry_run:
        if verbose:
            print(f"[execute_word] DRY RUN: would execute {len(segments)} segments from {filename}")
    else:
        if robot is None:
            raise ValueError("execute=True requires robot=urx.Robot(...) unless dry_run=True.")
        if (movej_fn is None) and (not hasattr(robot, "movej")):
            raise ValueError("execute=True requires movej_fn or robot.movej.")

    def issue_movej(q, tag=""):
        q = np.asarray(q, dtype=float).reshape(6,)

        if external_safety_filter is not None:
            # external filter expects classical joints. If pickle joints are already classical,
            # this is correct. If they're modified-DH, you must convert before filtering/executing.
            if not external_safety_filter(q):
                raise RuntimeError(f"[execute_word] Rejected by external safety filter {tag}")

        if dry_run:
            if verbose:
                print(f"[execute_word] movej{tag}: {np.array2string(q, precision=4)}")
            return

        if movej_fn is not None:
            movej_fn(q, **movej_kwargs)
        else:
            # Fallback: URX signature may vary. Prefer movej_fn.
            try:
                robot.movej(tuple(q.tolist()), **movej_kwargs)
            except TypeError as e:
                raise TypeError(
                    "[execute_word] robot.movej signature mismatch. Provide movej_fn "
                    "matching your Lab-1 urx calling convention."
                ) from e

    # Execute segments
    last_q = None
    for i, (theta_start, theta_end) in enumerate(segments):
        q0 = _as_1d_float_array(theta_start, expected_len=6)
        q1 = _as_1d_float_array(theta_end, expected_len=6)

        # Optional dedup to avoid redundant commands
        if (not dedup_consecutive) or (last_q is None) or (np.linalg.norm(q0 - last_q) > 1e-9):
            issue_movej(q0, tag=f" [seg {i} start]")
            last_q = q0.copy()

        if (not dedup_consecutive) or (np.linalg.norm(q1 - last_q) > 1e-9):
            issue_movej(q1, tag=f" [seg {i} end]")
            last_q = q1.copy()

    if verbose:
        print("[execute_word] done.")

    return None


# -----------------------------
# Angle utilities
# -----------------------------
def wrap_to_pi(x):
    """Wrap angles to (-pi, pi]."""
    x = np.asarray(x, dtype=float)
    return (x + np.pi) % (2.0 * np.pi) - np.pi


# -----------------------------
# Modified DH (Craig) helper
# -----------------------------
def dh_modified(a_im1: float, alpha_im1: float, d_i: float, theta_i: float) -> np.ndarray:
    """
    Craig Modified DH:
        T = Rx(alpha_{i-1}) * Tx(a_{i-1}) * Rz(theta_i) * Tz(d_i)

        [ ct,   -st,    0,    a_im1;
          st*ca, ct*ca, -sa, -d_i*sa;
          st*sa, ct*sa,  ca,  d_i*ca;
          0,      0,     0,    1 ]
    """
    ca = np.cos(alpha_im1)
    sa = np.sin(alpha_im1)
    ct = np.cos(theta_i)
    st = np.sin(theta_i)

    return np.array([
        [ct,      -st,     0.0,   a_im1],
        [st * ca, ct * ca, -sa,  -d_i * sa],
        [st * sa, ct * sa,  ca,   d_i * ca],
        [0.0,      0.0,    0.0,   1.0]
    ], dtype=float)


# -----------------------------
# Modified -> Classical joint variable conversion
# -----------------------------
def dh_modified_to_classical(thetaW) -> np.ndarray:
    """
    Convert Williams/modified DH 'thetaW' to classical UR joint angles qClass for FK.

    Your DHMODIFIEDTOCLASSICAL.m states:
      Williams home: thetaW = [0 0 0 0 0 0]
      Classical UR home: q = [0 -pi/2 0 -pi/2 0 0]

    Therefore:
      qClass = wrapToPi(thetaW + [0, -pi/2, 0, -pi/2, 0, 0])
    """
    thetaW = np.asarray(thetaW, dtype=float).reshape(-1)
    if thetaW.size != 6:
        raise ValueError("dh_modified_to_classical: thetaW must have length 6.")
    offset = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0], dtype=float)
    return wrap_to_pi(thetaW + offset)


def classical_to_dh_modified(qClass) -> np.ndarray:
    """
    Inverse of dh_modified_to_classical, used inside IK solver.

      thetaW = wrapToPi(qClass - offset)
    """
    qClass = np.asarray(qClass, dtype=float).reshape(-1)
    if qClass.size != 6:
        raise ValueError("classical_to_dh_modified: qClass must have length 6.")
    offset = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0], dtype=float)
    return wrap_to_pi(qClass - offset)


# -----------------------------
# SO(3) log map (rotation vector)
# -----------------------------
def so3_log(R: np.ndarray) -> np.ndarray:
    """
    Rotation matrix logarithm returning rotation vector w (axis * angle).
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError("so3_log: R must be 3x3.")

    tr = np.trace(R)
    c = (tr - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    th = float(np.arccos(c))

    if th < 1e-12:
        return np.zeros(3, dtype=float)

    s = np.sin(th)

    # Near pi: use diagonal-based axis extraction
    if abs(s) < 1e-12:
        axis = np.sqrt(np.maximum(0.0, (np.diag(R) + 1.0) / 2.0))
        axis = axis.astype(float)

        # Fix signs using off-diagonals
        if (R[2, 1] - R[1, 2]) < 0:
            axis[0] = -axis[0]
        if (R[0, 2] - R[2, 0]) < 0:
            axis[1] = -axis[1]
        if (R[1, 0] - R[0, 1]) < 0:
            axis[2] = -axis[2]

        return th * axis

    axis = (1.0 / (2.0 * s)) * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ], dtype=float)

    return th * axis

def is_duplicate(q: np.ndarray, Q: np.ndarray, tol=1e-3) -> bool:
    """
    True if q is within tol of any row of Q, using wrapped difference norm.
    """
    q = np.asarray(q, dtype=float).reshape(6,)
    if Q.size == 0:
        return False
    for k in range(Q.shape[0]):
        if np.linalg.norm(wrap_to_pi(q - Q[k, :])) < tol:
            return True
    return False

def ik(
    Tbt,
    T6t=None,
    thetaSeed=None,
    fk_func=None,
    safety_check_func=None,
    joint_limits_rad=None,
    joint_limit_margin_rad=0.0,
):
    """
    Analytical IK for UR10e (classical DH), returning Williams/modified theta variables.

    Pipeline:
      1) Convert desired tool pose Tbt into desired flange pose T_B6 by removing T6t.
      2) Closed-form UR IK to generate up to 8 classical joint solutions q_class.
      3) Guardrails: joint limits (+ optional margin) and optional safety_check_func.
      4) Convert surviving classical solutions -> Williams/modified (theta_all).
      5) Choose thetaIK closest to thetaSeed (wrapped distance).

    Returns
    -------
    thetaIK : (6,)
        Chosen solution in Williams/modified theta variables.
    thetaAll : (M,6)
        All candidate solutions in Williams/modified theta variables.
    info : dict
        Metadata and diagnostics.
    """
    if fk_func is None:
        raise ValueError("ik: you must pass fk_func (the fk function).")

    Tbt = np.asarray(Tbt, dtype=float)
    if Tbt.shape != (4, 4):
        raise ValueError("ik: Tbt must be 4x4.")

    if T6t is None:
        T6t = np.eye(4, dtype=float)
    else:
        T6t = np.asarray(T6t, dtype=float)
        if T6t.shape != (4, 4):
            raise ValueError("ik: T6t must be 4x4.")

    if thetaSeed is None:
        thetaSeed = np.zeros(6, dtype=float)
    thetaSeed = np.asarray(thetaSeed, dtype=float).reshape(6,)

    # -------------------------
    # Joint limits guardrail
    # -------------------------
    if joint_limits_rad is None:
        joint_limits_rad = [(-2.0 * np.pi, 2.0 * np.pi)] * 6
    if len(joint_limits_rad) != 6:
        raise ValueError("ik: joint_limits_rad must be length-6 list of (lo, hi).")

    def within_limits(q):
        q = np.asarray(q, dtype=float).reshape(6,)
        for i in range(6):
            lo, hi = joint_limits_rad[i]
            if not (lo <= q[i] <= hi):
                return False
            if joint_limit_margin_rad > 0:
                if (q[i] < lo + joint_limit_margin_rad) or (q[i] > hi - joint_limit_margin_rad):
                    return False
        return True

    def clamp(x, lo=-1.0, hi=1.0):
        return float(np.clip(x, lo, hi))

    def inv_T(T):
        R = T[:3, :3]
        p = T[:3, 3]
        Ti = np.eye(4, dtype=float)
        Ti[:3, :3] = R.T
        Ti[:3, 3] = -R.T @ p
        return Ti

    # -------------------------
    # UR10e classical DH parameters
    # -------------------------
    a = np.array([0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0], dtype=float)
    d = np.array([0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655], dtype=float)
    alpha = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0], dtype=float)

    # -------------------------
    # Remove tool offset: T_B6_des = Tbt * inv(T6t)
    # -------------------------
    T_B6_des = Tbt @ inv_T(T6t)
    R06 = T_B6_des[:3, :3]
    p06 = T_B6_des[:3, 3]

    # -------------------------
    # Closed-form UR IK (returns classical q solutions)
    # -------------------------
    q_sols = []
    sol_details = []

    # wrist center of frame 5 (origin of frame 5): p05 = p06 - d6 * z6
    p05 = p06 - d[5] * R06[:, 2]
    px, py = float(p05[0]), float(p05[1])
    r_xy = float(np.hypot(px, py))

    # If the wrist center is too close to base axis, no solution
    if r_xy < abs(d[3]) - 1e-12:
        thetaIK = np.full(6, np.nan, dtype=float)
        thetaAll = np.zeros((0, 6), dtype=float)
        info = {
            "success": False,
            "message": "ik: no solution (wrist center too close to base axis for q1).",
            "thetaAll": thetaAll,
            "qAll_classical": np.zeros((0, 6), dtype=float),
            "solutions": [],
        }
        return thetaIK, thetaAll, info

    # q1 candidates
    phi = float(np.arctan2(py, px))
    delta = float(np.arccos(clamp(d[3] / r_xy)))
    q1_candidates = [phi + delta + np.pi / 2, phi - delta + np.pi / 2]

    for q1 in q1_candidates:
        q1 = float(wrap_to_pi(q1))

        # q5 candidates from geometry:
        # q5 = ± acos( (p06_x*sin(q1) - p06_y*cos(q1) - d4) / d6 )
        arg5 = (p06[0] * np.sin(q1) - p06[1] * np.cos(q1) - d[3]) / d[5]
        arg5 = clamp(arg5)
        q5_base = float(np.arccos(arg5))
        q5_candidates = [q5_base, -q5_base]

        for q5 in q5_candidates:
            q5 = float(wrap_to_pi(q5))

            s5 = float(np.sin(q5))
            if abs(s5) < 1e-8:
                # Wrist singular; skip for simplicity (can be handled, but not needed so far)
                continue

            # q6 from rotation
            q6 = float(np.arctan2(
                (-R06[0, 1] * np.sin(q1) + R06[1, 1] * np.cos(q1)) / s5,
                ( R06[0, 0] * np.sin(q1) - R06[1, 0] * np.cos(q1)) / s5
            ))
            q6 = float(wrap_to_pi(q6))

            # Build T14 = inv(T01)*T06*inv(T56)*inv(T45)
            T01 = dh_classical(a[0], alpha[0], d[0], q1)
            T10 = inv_T(T01)

            T56 = dh_classical(a[5], alpha[5], d[5], q6)
            T65 = inv_T(T56)

            T45 = dh_classical(a[4], alpha[4], d[4], q5)
            T54 = inv_T(T45)

            T14 = T10 @ T_B6_des @ T65 @ T54
            p14 = T14[:3, 3]

            # Key UR fact (given alpha1=+90°): solve planar 2-link in (x1,y1) plane, not (x1,z1)
            x = float(p14[0])
            y = float(p14[1])
            r2 = x * x + y * y

            # q3 from law of cosines
            cos3 = (r2 - a[1] ** 2 - a[2] ** 2) / (2.0 * a[1] * a[2])
            if cos3 < -1.0 - 1e-8 or cos3 > 1.0 + 1e-8:
                continue
            cos3 = clamp(cos3)
            q3_base = float(np.arccos(cos3))
            q3_candidates = [q3_base, -q3_base]

            for q3 in q3_candidates:
                q3 = float(wrap_to_pi(q3))

                # q2
                q2 = float(np.arctan2(y, x) - np.arctan2(a[2] * np.sin(q3), a[1] + a[2] * np.cos(q3)))
                q2 = float(wrap_to_pi(q2))

                # q4: since R14 = Rz(q2+q3+q4) * Rx(pi/2), we extract q234 from atan2(R14[1,0], R14[0,0])
                q234 = float(np.arctan2(T14[1, 0], T14[0, 0]))
                q4 = float(wrap_to_pi(q234 - q2 - q3))

                q_sol = wrap_to_pi(np.array([q1, q2, q3, q4, q5, q6], dtype=float))

                # Guardrail 1: joint limits
                ok = within_limits(q_sol)

                # Guardrail 2: internal safety check (table collision etc.)
                if ok and (safety_check_func is not None):
                    try:
                        safe = safety_check_func(q_sol, T6t)
                    except TypeError:
                        safe = safety_check_func(q_sol)
                    if not safe:
                        ok = False

                if ok:
                    if len(q_sols) == 0 or (not is_duplicate(q_sol, np.vstack(q_sols), tol=1e-3)):
                        q_sols.append(q_sol)
                        sol_details.append({"branch": "analytic", "q": q_sol.copy()})

    if len(q_sols) == 0:
        thetaIK = np.full(6, np.nan, dtype=float)
        thetaAll = np.zeros((0, 6), dtype=float)
        info = {
            "success": False,
            "message": "ik: no solution found (analytic branches invalid / rejected by guardrails).",
            "thetaAll": thetaAll,
            "qAll_classical": np.zeros((0, 6), dtype=float),
            "solutions": sol_details,
        }
        return thetaIK, thetaAll, info

    q_all = np.vstack(q_sols)
    theta_all = np.vstack([classical_to_dh_modified(q_all[k, :]) for k in range(q_all.shape[0])])

    # Choose solution closest to thetaSeed (wrapped distance in modified/Williams space)
    dists = np.array(
        [np.linalg.norm(wrap_to_pi(theta_all[k, :] - thetaSeed)) for k in range(theta_all.shape[0])],
        dtype=float
    )
    idx = int(np.argmin(dists))
    thetaIK = theta_all[idx, :].copy()

    info = {
        "success": True,
        "message": f"ik: analytic found {theta_all.shape[0]} solution(s).",
        "thetaAll": theta_all,
        "qAll_classical": q_all,
        "solutions": sol_details,
        "chosen_index": idx,
        "joint_limits_rad": joint_limits_rad,
        "joint_limit_margin_rad": float(joint_limit_margin_rad),
    }
    return thetaIK, theta_all, info


def set_initial_pose(
    Tbt_init,
    *,
    T6t=None,
    thetaSeed=None,                 # modified-DH seed
    fk_func=None,
    safety_check_func=None,
    external_safety_filter=None,    # TA filter on CLASSICAL joints
    joint_limits_rad=None,
    joint_limit_margin_rad=0.0,
    # execution controls
    simulate=True,                  # True => no motion commands (but still solve IK + report)
    dry_run=False,                  # if simulate=False and dry_run=True => no motion commands
    robot=None,
    movej_fn=None,
    movej_kwargs=None,              # forwarded to movej_fn(...) or robot.movej(...)
    # verification
    verify_fk=True,
    pos_tol=1e-4,                   # meters
    rot_tol=1e-4,                   # radians (norm of so3_log)
    on_fail="raise",                # "raise" | "return"
):
    """
    Move UR10e to a user-specified initial tool pose Tbt_init (base->tool).

    Pipeline:
      1) IK in modified DH angles (theta_mod)
      2) Convert to classical (q_class)
      3) Apply external safety filter (required) on q_class
      4) Optional in-code safety_check_func on theta_mod (if your IK didn’t already)
      5) Command robot via movej_fn (preferred) or robot.movej fallback
      6) Optional FK verification: fk(q_class,T6t) ~ Tbt_init

    Returns a dict with:
      theta_mod, q_class, info, (optional) verification errors.
    """
    if fk_func is None:
        raise ValueError("set_initial_pose: pass fk_func=fk")
    if movej_kwargs is None:
        movej_kwargs = {}
    if T6t is None:
        T6t = np.eye(4, dtype=float)

    Tbt_init = np.asarray(Tbt_init, dtype=float)
    if Tbt_init.shape != (4, 4):
        raise ValueError("set_initial_pose: Tbt_init must be 4x4.")

    if thetaSeed is None:
        thetaSeed = np.zeros(6, dtype=float)
    else:
        thetaSeed = np.asarray(thetaSeed, dtype=float).reshape(6,)

    # -------------------------
    # 1) Solve IK (modified DH)
    # -------------------------
    theta_mod, theta_all, info = ik(
        Tbt_init,
        T6t=T6t,
        thetaSeed=thetaSeed,
        fk_func=fk_func,
        safety_check_func=safety_check_func,
        joint_limits_rad=joint_limits_rad,
        joint_limit_margin_rad=joint_limit_margin_rad,
    )

    if (not info.get("success", False)) or (theta_mod is None) or np.any(np.isnan(theta_mod)):
        msg = info.get("message", "IK failed.")
        if on_fail == "raise":
            raise RuntimeError(f"set_initial_pose: {msg}")
        return {"success": False, "info": info, "theta_mod": theta_mod, "q_class": None}

    theta_mod = np.asarray(theta_mod, dtype=float).reshape(6,)

# ===================== TA: SAFETY FILTER FOR INITIAL POSE =====================
# After IK -> classical conversion, run external_safety_filter(q_class) before any moveJ.
# =============================================================================

    # -------------------------
    # 2) Convert to classical
    # -------------------------
    q_class = dh_modified_to_classical(theta_mod)
    q_class = np.asarray(q_class, dtype=float).reshape(6,)

    # -------------------------
    # 3) External safety filter (CLASSICAL joints)
    # -------------------------
    if external_safety_filter is not None:
        ok = external_safety_filter(q_class)
        if not ok:
            msg = "Rejected by external safety filter."
            if on_fail == "raise":
                raise RuntimeError(f"set_initial_pose: {msg}")
            return {
                "success": False,
                "info": {**info, "success": False, "message": msg},
                "theta_mod": theta_mod,
                "q_class": q_class,
            }

    # -------------------------
    # 4) Optional extra safety check in modified space
    #    (Only if caller provided it; IK already uses it if passed.)
    # -------------------------
    if safety_check_func is not None:
        try:
            if not safety_check_func(theta_mod, T6t):
                msg = "Rejected by safety_check_func."
                if on_fail == "raise":
                    raise RuntimeError(f"set_initial_pose: {msg}")
                return {
                    "success": False,
                    "info": {**info, "success": False, "message": msg},
                    "theta_mod": theta_mod,
                    "q_class": q_class,
                }
        except TypeError:
            # if users implements safety_check(theta_mod) without T6t; tolerate both.
            if not safety_check_func(theta_mod):
                msg = "Rejected by safety_check_func."
                if on_fail == "raise":
                    raise RuntimeError(f"set_initial_pose: {msg}")
                return {
                    "success": False,
                    "info": {**info, "success": False, "message": msg},
                    "theta_mod": theta_mod,
                    "q_class": q_class,
                }

    out = {
        "success": True,
        "info": info,
        "theta_mod": theta_mod,
        "theta_all": theta_all,
        "q_class": q_class,
    }

    # -------------------------
    # 5) Execute (optional)
    # -------------------------
    if not simulate:
        if dry_run:
            out["executed"] = False
            out["dry_run"] = True
        else:
            if robot is None:
                raise ValueError("set_initial_pose: robot must be provided when simulate=False and dry_run=False.")
            # Prefer a movej_fn that matches your Lab-1 urx signature.
            if movej_fn is not None:
                movej_fn(q_class, **movej_kwargs)
            else:
                if not hasattr(robot, "movej"):
                    raise ValueError("set_initial_pose: no movej_fn provided and robot has no .movej method.")
                try:
                    robot.movej(q_class, **movej_kwargs)
                except TypeError as e:
                    raise TypeError(
                        "set_initial_pose: robot.movej signature mismatch. "
                        "Provide movej_fn that matches your Lab-1 URX calling convention."
                    ) from e
            out["executed"] = True
            out["dry_run"] = False

    # -------------------------
    # 6) Optional FK verification
    # -------------------------
    if verify_fk:
        Tcheck = fk_func(q_class, T6t)
        pos_err = float(np.linalg.norm(Tcheck[:3, 3] - Tbt_init[:3, 3]))
        R_err = Tcheck[:3, :3].T @ Tbt_init[:3, :3]
        rot_err = float(np.linalg.norm(so3_log(R_err)))
        out["verify"] = {
            "pos_err_m": pos_err,
            "rot_err_rad": rot_err,
            "pos_tol_m": float(pos_tol),
            "rot_tol_rad": float(rot_tol),
            "ok": (pos_err <= pos_tol and rot_err <= rot_tol),
        }

        if (not out["verify"]["ok"]) and (on_fail == "raise"):
            raise RuntimeError(
                "set_initial_pose: FK verification failed "
                f"(pos_err={pos_err:.3e} m, rot_err={rot_err:.3e} rad)."
            )

    return out



# -----------------------------
# Optional: pipeline validation (debug helper)
# -----------------------------
def validate_ik_pipeline(N=25, fk_func=None, safety_check_func=None, rng_seed=0):
    """
    Samples random Williams thetaTrue, maps -> classical -> FK -> IK -> classical -> FK,
    reports mean/max position and rotation error.
    """
    if fk_func is None:
        raise ValueError("validate_ik_pipeline: pass fk_func (your fk).")

    rng = np.random.default_rng(rng_seed)

    T6t = np.eye(4, dtype=float)

    pos_errs = []
    rot_errs = []
    fails = 0

    for k in range(N):
        theta_true = (rng.random(6) - 0.5) * (2.0 * np.pi / 3.0)
        q_true = dh_modified_to_classical(theta_true)
        Tbt = fk_func(q_true, T6t)

        theta_seed = np.zeros(6, dtype=float)
        theta_ik, _, info = ik(Tbt, T6t=T6t, thetaSeed=theta_seed, fk_func=fk_func, safety_check_func=safety_check_func)

        if (not info["success"]) or np.any(np.isnan(theta_ik)):
            fails += 1
            continue

        q_hat = dh_modified_to_classical(theta_ik)
        T_hat = fk_func(q_hat, T6t)

        pos_err = float(np.linalg.norm(Tbt[:3, 3] - T_hat[:3, 3]))

        R = Tbt[:3, :3].T @ T_hat[:3, :3]
        c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
        rot_err = float(np.arccos(c))

        pos_errs.append(pos_err)
        rot_errs.append(rot_err)

    pos_errs = np.array(pos_errs) if len(pos_errs) else np.array([np.nan])
    rot_errs = np.array(rot_errs) if len(rot_errs) else np.array([np.nan])

    print(f"\nRound-trip validation over N={N} tests:")
    print(f"Position error (m): mean {np.nanmean(pos_errs):.3e}, max {np.nanmax(pos_errs):.3e}")
    print(f"Rotation error (rad): mean {np.nanmean(rot_errs):.3e}, max {np.nanmax(rot_errs):.3e}")
    print(f"Failures: {fails} / {N}")



def _tool_orientation_not_axis_aligned():
    """
    Returns a fixed rotation matrix for the tool frame that points roughly "down"
    but with a small tilt so it's not axis-aligned with the base (helps IK stability).
    """
    def Rx(a):
        ca, sa = np.cos(a), np.sin(a)
        return np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]], float)
    def Ry(a):
        ca, sa = np.cos(a), np.sin(a)
        return np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]], float)
    def Rz(a):
        ca, sa = np.cos(a), np.sin(a)
        return np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]], float)

    return Rz(np.deg2rad(35.0)) @ Ry(np.pi) @ Rx(np.deg2rad(10.0))


def _pose_from_xy(x, y, z, R=None):
    """Build 4x4 pose with translation (x,y,z) and rotation R."""
    if R is None:
        R = _tool_orientation_not_axis_aligned()
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def _seven_seg_strokes(ch, w=0.03, h=0.06):
    """
    Returns a list of strokes, each stroke is a list of 2D points [(x,y), ...]
    in local character coordinates with origin at lower-left.

    This is intentionally simple. It is acceptable per spec that some letters
    (e.g. u and v) may look similar.
    """
    ch = ch.upper()
    # Segment endpoints in (x,y) with width w and height h
    A = [(0, h), (w, h)]           # top
    B = [(w, h), (w, h/2)]         # upper right
    C = [(w, h/2), (w, 0)]         # lower right
    D = [(0, 0), (w, 0)]           # bottom
    E = [(0, h/2), (0, 0)]         # lower left
    F = [(0, h), (0, h/2)]         # upper left
    G = [(0, h/2), (w, h/2)]       # middle

    segs = {"A": A, "B": B, "C": C, "D": D, "E": E, "F": F, "G": G}

    # A crude mapping (enough to test pipeline)
    map7 = {
        "0": ["A","B","C","D","E","F"],
        "1": ["B","C"],
        "2": ["A","B","G","E","D"],
        "3": ["A","B","G","C","D"],
        "4": ["F","G","B","C"],
        "5": ["A","F","G","C","D"],
        "6": ["A","F","G","E","C","D"],
        "7": ["A","B","C"],
        "8": ["A","B","C","D","E","F","G"],
        "9": ["A","B","C","D","F","G"],
        "H": ["F","E","G","B","C"],
        "E": ["A","F","G","E","D"],
        "L": ["F","E","D"],
        "O": ["A","B","C","D","E","F"],
        "U": ["F","E","C","B","D"],
        "T": ["A","G"],   # not great
        "R": ["A","F","G","B","E","C"],  # not great
        "P": ["A","F","G","B","E"],      # not great
        " ": [],
        "-": ["G"],
    }

    chosen = map7.get(ch, map7.get(" ", []))

    # Each segment is its own stroke (2-point polyline)
    strokes = []
    for s in chosen:
        strokes.append(segs[s])
    return strokes


def draw_character(
    Tbt_start,
    ch,
    T6t=None,
    z0=0.10,
    z_touch=0.02,
    n_per_segment=8,
    fk_func=None,
    safety_check_func=None,
    external_safety_filter=None,
    thetaSeed=None,
    joint_limits_rad=None,
    joint_limit_margin_rad=0.0,
    simulate=True,
    robot=None,
    translate_fn=None,
    char_w=0.03,
    char_h=0.06,
    on_fail="skip",      # "skip" | "raise"
    dry_run=False,       # if True and simulate=False, do not move robot; just track p_end
    p_end=None,          # np.array shape (3,) initial p_end (optional)
    print_p_end=False,   # print p_end at each "command"
    p_end_label="",      # prefix label for printing
    movej_fn=None,       # preferred: wrapper matching Lab-1 urx signature
    movej_kwargs=None,   # forwarded to movej_fn(...) or robot.movej(...)
):
    """
    Plans a single character as 7-seg strokes.

    Execution behavior:
      - simulate=True  : no robot commands (planning only)
      - simulate=False :
          - dry_run=True  : emulate motion and track p_end (no robot commands)
          - dry_run=False : REAL hardware:
              1) moveJ to the first IK solution (q_class_list[0])
              2) translate between consecutive planned poses (base-frame deltas)

    Returns dict including:
      - poses, q_class, theta_mod, pen_down
      - success, failures
      - p_end_final (if dry_run)
    """
    if fk_func is None:
        raise ValueError("draw_character: pass fk_func=fk")
    if movej_kwargs is None:
        movej_kwargs = {}

    if T6t is None:
        T6t = np.eye(4, dtype=float)

    Tbt_start = np.asarray(Tbt_start, dtype=float)
    if Tbt_start.shape != (4, 4):
        raise ValueError("draw_character: Tbt_start must be 4x4.")

    x0, y0, _ = Tbt_start[:3, 3]
    R = _tool_orientation_not_axis_aligned()

    if thetaSeed is None:
        thetaSeed = np.zeros(6, dtype=float)
    thetaSeed = np.asarray(thetaSeed, dtype=float).reshape(6,)

    # Case-sensitive stroke generator
    strokes = seven_seg_strokes_any_letter(ch, w=char_w, h=char_h)

    poses, q_class_list, theta_mod_list = [], [], []
    pen_down_flags = []
    failures = []

    def solve_pose(T):
        theta_mod, _, info = ik(
            T, T6t=T6t, thetaSeed=thetaSeed,
            fk_func=fk_func, safety_check_func=safety_check_func,
            joint_limits_rad=joint_limits_rad,
            joint_limit_margin_rad=joint_limit_margin_rad,
        )
        if (not info.get("success", False)) or np.any(np.isnan(theta_mod)):
            return None, None, info

# =========================================================================================
# TA CHECKPOINT (In lab part d):
# IK returns modified-DH angles -> convert to CLASSICAL UR joints -> run external safety filter.
# Convert before sending joint angles to the robot.
# See external_safety_filter(q_class) below.
# =========================================================================================

        q_class = dh_modified_to_classical(theta_mod)

        # External safety filter is required before commanding robot.
        if external_safety_filter is not None:
            if not external_safety_filter(q_class):
                info = dict(info)
                info["success"] = False
                info["message"] = "Rejected by external safety filter."
                return None, None, info

        return theta_mod, q_class, info

    def append_pose(T, pen_down: bool):
        nonlocal thetaSeed
        th, qc, info = solve_pose(T)
        if th is None:
            failures.append({"pose": T.copy(), "info": info})
            if on_fail == "raise":
                raise RuntimeError(f"draw_character: IK failed for '{ch}': {info.get('message','')}")
            return False
        poses.append(T)
        theta_mod_list.append(th)
        q_class_list.append(qc)
        pen_down_flags.append(bool(pen_down))
        thetaSeed = th.copy()
        return True


# ============================================================
# TA CHECKPOINT (In lab part d):
#   - hover at origin: append_pose(..., z0)     
#   - approach down:   append_pose(..., z_touch)
#   - stroke at touch: append_pose(..., z_touch)
#   - lift back up:    append_pose(..., z0)     
# ============================================================


    # Initial pen-up pose at character origin
    if not append_pose(_pose_from_xy(x0, y0, z0, R), pen_down=False):
        return {
            "poses": [], "q_class": [], "theta_mod": [], "pen_down": [],
            "failures": failures, "success": False, "p_end_final": p_end
        }

    # Draw each segment: pen-up to start, down, draw, up
    for seg in strokes:
        (sx, sy), (ex, ey) = seg

        if not append_pose(_pose_from_xy(x0 + sx, y0 + sy, z0, R), pen_down=False):
            continue
        if not append_pose(_pose_from_xy(x0 + sx, y0 + sy, z_touch, R), pen_down=True):
            continue

        for t in np.linspace(0.0, 1.0, n_per_segment):
            x = x0 + (1 - t) * sx + t * ex
            y = y0 + (1 - t) * sy + t * ey
            if not append_pose(_pose_from_xy(x, y, z_touch, R), pen_down=True):
                break

        append_pose(_pose_from_xy(x0 + ex, y0 + ey, z0, R), pen_down=False)

    out = {
        "poses": poses,
        "q_class": q_class_list,
        "theta_mod": theta_mod_list,
        "pen_down": pen_down_flags,
        "failures": failures,
        "success": (len(failures) == 0),
        "char": ch,
        "origin_xy": (x0, y0),
    }

    # -------------------------
    # EXECUTION / DRY-RUN BLOCK
    # -------------------------
    if not simulate:
        # Guard: no planned poses means no execution
        if len(poses) == 0:
            out["executed"] = False
            return out

        if dry_run:
            # Initialize p_end
            if p_end is None:
                T_fk0 = fk_func(q_class_list[0], T6t)
                p_end = T_fk0[:3, 3].copy()
            else:
                p_end = np.asarray(p_end, dtype=float).reshape(3,)

            def _print(step, pen_state):
                if print_p_end:
                    print(f"{p_end_label}{ch} step={step:03d} pen={'DOWN' if pen_state else 'UP  '} p_end={p_end}")

            # "Command" 0: moveJ to first pose (emulated by FK)
            T_fk0 = fk_func(q_class_list[0], T6t)
            p_end = T_fk0[:3, 3].copy()
            _print(0, pen_down_flags[0])

            # Subsequent "commands": emulate translate by Δp between consecutive planned poses
            for k in range(1, len(poses)):
                dp = poses[k][:3, 3] - poses[k - 1][:3, 3]
                p_end = p_end + dp
                _print(k, pen_down_flags[k])

            out["p_end_final"] = p_end.copy()
            out["executed"] = False
            return out

# ===========================================================================
# TA CHECKPOINT (In lab part d):
# We first moveJ to the first planned IK solution (q0) before any translate commands.
# This enablse positioning the robot before drawing.
# ===========================================================================

        # REAL robot execution
        if robot is None or translate_fn is None:
            raise ValueError("draw_character: for hardware pass robot and translate_fn, or set dry_run=True.")

        # 1) moveJ to the first IK pose (critical!)
        q0 = q_class_list[0]
        if external_safety_filter is not None:
            # Redundant with planning-time filter, but cheap + safer.
            if not external_safety_filter(q0):
                raise RuntimeError("draw_character: first pose rejected by external safety filter at execution time.")

        if movej_fn is not None:
            movej_fn(q0, **movej_kwargs)
        else:
            # Fallback: try robot.movej, but this may be URX-version dependent.
            if not hasattr(robot, "movej"):
                raise ValueError("draw_character: no movej_fn provided and robot has no .movej method.")
            try:
                robot.movej(q0, **movej_kwargs)
            except TypeError as e:
                raise TypeError(
                    "draw_character: robot.movej signature mismatch. "
                    "Provide movej_fn that matches your Lab-1 URX calling convention."
                ) from e

# ============================================================
# TA CHECKPOINT (In lab part d):
# CARTESIAN EXECUTION VIA TRANSLATE:
# We execute the character by translating between consecutive planned poses.
# dp is a base-frame Cartesian delta; translate_fn sends rob.translate(dx,dy,dz).
# ============================================================


        # 2) translate between consecutive planned poses
        for k in range(1, len(poses)):
            dp = poses[k][:3, 3] - poses[k - 1][:3, 3]
            translate_fn(robot, float(dp[0]), float(dp[1]), float(dp[2]))

        out["executed"] = True

    return out



def compute_max_chars_per_line(max_width, char_w, char_spacing):
    """
    Returns the maximum number of characters that fit in max_width.
    """
    advance = char_w + char_spacing
    if advance <= 0:
        return 0
    return int(np.floor(max_width / advance))



def draw_string(
    s,
    *,
    Tboard,
    T6t=None,
    z0=0.10,
    z_touch=0.02,
    char_w=0.03,
    char_h=0.06,
    char_spacing=0.01,
    line_spacing=0.02,
    n_per_segment=8,
    fk_func=None,
    safety_check_func=None,
    external_safety_filter=None,
    joint_limits_rad=None,
    joint_limit_margin_rad=0.0,
    thetaSeed0=None,
    simulate=True,
    robot=None,
    translate_fn=None,
    on_fail="skip",            # "skip" | "raise"
    dry_run=False,             # if True and simulate=False, do not move robot; just track p_end
    move_to_first_pose=True,
    movej_fn=None,
    movej_kwargs=None,
    print_p_end=False,
):
    """
    Draw a string on the whiteboard using your 7-seg stroke planner.

    Execution modes:
      - simulate=True: plan only, no robot commands.
      - simulate=False:
          - dry_run=True : no commands; emulate motion and track p_end.
          - dry_run=False: command UR10e:
              * optionally moveJ to the first pose BEFORE drawing (move_to_first_pose=True)
              * for each character, draw_character will:
                    (1) moveJ to that character's first pose
                    (2) translate between consecutive poses

    Returns: dict with per-character outputs and overall success.
    """
    if fk_func is None:
        raise ValueError("draw_string: pass fk_func=fk")
    if movej_kwargs is None:
        movej_kwargs = {}
    if T6t is None:
        T6t = np.eye(4, dtype=float)

    Tboard = np.asarray(Tboard, dtype=float)
    if Tboard.shape != (4, 4):
        raise ValueError("draw_string: Tboard must be 4x4.")

    # Cursor in BOARD coordinates (meters)
    x_cursor = 0.0
    y_cursor = 0.0

    # Seed for IK continuity (modified DH angles)
    if thetaSeed0 is None:
        thetaSeed = np.zeros(6, dtype=float)
    else:
        thetaSeed = np.asarray(thetaSeed0, dtype=float).reshape(6,)

    # Helper: board(x,y,z) -> base frame position (with board axes embedded in Tboard)
    def board_xyz_to_base(x, y, z):
        p = Tboard @ np.array([x, y, z, 1.0], dtype=float)
        return p[:3]

    # Helper: build start pose in BASE for a character origin at (x_cursor, y_cursor)
    def make_Tbt_start(xb, yb, z):
        R = _tool_orientation_not_axis_aligned()
        p_base = board_xyz_to_base(xb, yb, 0.0)  # board origin plane; z handled separately in base
        # IMPORTANT: interpret z as base-frame Z offset, consistent with the rest of your code
        # (to get z along board-normal instead, change this to p_base + z * Tboard[:3,2])
        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        T[:3, 3] = np.array([p_base[0], p_base[1], z], dtype=float)
        return T

    # Identify first drawable character (used if move_to_first_pose=True)
    def is_drawable(ch):
        return (ch not in [" ", "\t", "\n", "\r"])

    first_drawable = next((ch for ch in s if is_drawable(ch)), None)

# ===================== TA (part 2d): MOVE TO START POSE (BEFORE DRAWING) =====================
# If simulate=False and dry_run=False and move_to_first_pose=True:
#   - compute first character start pose from Tboard + cursor
#   - run IK
#   - apply external safety filter
#   - moveJ robot to start pose BEFORE drawing begins
# =============================================================================================

    # ---- move-to-first-pose happens BEFORE drawing ----
    if (not simulate) and (not dry_run) and move_to_first_pose and (first_drawable is not None):
        Tbt_first = make_Tbt_start(x_cursor, y_cursor, z0)

        theta_mod_first, _, info = ik(
            Tbt_first,
            T6t=T6t,
            thetaSeed=thetaSeed,
            fk_func=fk_func,
            safety_check_func=safety_check_func,
            joint_limits_rad=joint_limits_rad,
            joint_limit_margin_rad=joint_limit_margin_rad,
        )
        if (not info.get("success", False)) or np.any(np.isnan(theta_mod_first)):
            msg = info.get("message", "")
            if on_fail == "raise":
                raise RuntimeError(f"draw_string: IK failed for first pose: {msg}")
            # If skipping, we just won't move to first pose.
        else:
            q_class_first = dh_modified_to_classical(theta_mod_first)

            if external_safety_filter is not None and (not external_safety_filter(q_class_first)):
                if on_fail == "raise":
                    raise RuntimeError("draw_string: first pose rejected by external safety filter.")
            else:
                if robot is None:
                    raise ValueError("draw_string: robot must be provided when simulate=False and dry_run=False.")
                if movej_fn is not None:
                    movej_fn(q_class_first, **movej_kwargs)
                else:
                    # Fallback: may be URX-version dependent.
                    if not hasattr(robot, "movej"):
                        raise ValueError("draw_string: no movej_fn provided and robot has no .movej method.")
                    try:
                        robot.movej(q_class_first, **movej_kwargs)
                    except TypeError as e:
                        raise TypeError(
                            "draw_string: robot.movej signature mismatch. "
                            "Provide movej_fn that matches your Lab-1 URX calling convention."
                        ) from e

                # Seed continuity: after moving, update seed to the solution we used
                thetaSeed = theta_mod_first.copy()

    # For dry_run mode, track an emulated end-effector position p_end in base frame
    p_end = None

    char_outputs = []
    overall_failures = []
    success = True

    for ch in s:
        if ch in ["\r"]:
            continue

        if ch == "\n":
            # New line: reset x, decrement y
            x_cursor = 0.0
            y_cursor -= (char_h + line_spacing)
            continue

        if ch in [" ", "\t"]:
            # Space/tab: advance cursor
            x_cursor += (char_w + char_spacing) * (4.0 if ch == "\t" else 1.0)
            continue

        # Build this character's BASE start pose
        Tbt_start = make_Tbt_start(x_cursor, y_cursor, z0)

        # Draw character
        out_ch = draw_character(
            Tbt_start,
            ch,
            T6t=T6t,
            z0=z0,
            z_touch=z_touch,
            n_per_segment=n_per_segment,
            fk_func=fk_func,
            safety_check_func=safety_check_func,
            external_safety_filter=external_safety_filter,
            thetaSeed=thetaSeed,
            joint_limits_rad=joint_limits_rad,
            joint_limit_margin_rad=joint_limit_margin_rad,
            simulate=simulate,
            robot=robot,
            translate_fn=translate_fn,
            char_w=char_w,
            char_h=char_h,
            on_fail=on_fail,
            dry_run=dry_run,
            p_end=p_end,
            print_p_end=print_p_end,
            p_end_label="[draw_string] ",
            movej_fn=movej_fn,
            movej_kwargs=movej_kwargs,
        )

        char_outputs.append(out_ch)

        # Update seed continuity if that character produced any solutions
        if out_ch.get("theta_mod", None) and len(out_ch["theta_mod"]) > 0:
            thetaSeed = np.asarray(out_ch["theta_mod"][-1], dtype=float).reshape(6,)

        # Update dry-run p_end
        if dry_run and ("p_end_final" in out_ch) and (out_ch["p_end_final"] is not None):
            p_end = np.asarray(out_ch["p_end_final"], dtype=float).reshape(3,)

        # Track failures
        if not out_ch.get("success", True):
            success = False
            overall_failures.extend(out_ch.get("failures", []))
            if on_fail == "raise":
                # draw_character would have raised already; this is just defensive.
                raise RuntimeError(f"draw_string: character '{ch}' failed.")

        # Advance cursor after drawing a character
        x_cursor += (char_w + char_spacing)

    return {
        "string": s,
        "characters": char_outputs,
        "failures": overall_failures,
        "success": success,
        "p_end_final": p_end,
    }



def seven_seg_strokes_any_letter(ch, w=0.03, h=0.06):
    """
    Return strokes for any A-Z / a-z / 0-9 / space / dash using a 7-segment style.

    Output format:
      strokes = [ [(x0,y0),(x1,y1)], ... ]  each is one straight segment.
    """
    if not isinstance(ch, str) or len(ch) == 0:
        ch = " "
    chU = ch.upper()

    # Segment endpoints in local coords: origin at lower-left
    A = [(0, h),   (w, h)]     # top
    B = [(w, h),   (w, h/2)]   # upper right
    C = [(w, h/2), (w, 0)]     # lower right
    D = [(0, 0),   (w, 0)]     # bottom
    E = [(0, h/2), (0, 0)]     # lower left
    F = [(0, h),   (0, h/2)]   # upper left
    G = [(0, h/2), (w, h/2)]   # middle

    segs = {"A": A, "B": B, "C": C, "D": D, "E": E, "F": F, "G": G}

    # Canonical 7-seg codes (not typographically perfect; but covers A–Z)
    # Convention: list of segments to "turn on"
    codes = {
        "0": "AB CDEF".replace(" ", ""),
        "1": "BC",
        "2": "ABGED",
        "3": "ABGCD",
        "4": "FGBC",
        "5": "AFGCD",
        "6": "AFGECD",
        "7": "ABC",
        "8": "ABCDEFG",
        "9": "AFGBCD",

        "A": "ABCEFG",
        "B": "FGCDE",      # looks like b
        "C": "ADEF",
        "D": "BCDEG",      # not great
        "E": "AFGED",
        "F": "AFGE",
        "G": "A F ECD".replace(" ", ""),  # A+F+E+C+D (no mid) -> "G-ish"
        "H": "FGEBC",
        "I": "BC",         # like 1
        "J": "BCD",
        "K": "FGEBC",      # fallback (K not great in 7-seg)
        "L": "DEF",
        "M": "ABCEFG",     # fallback similar to A
        "N": "FGEBC",      # fallback similar to H
        "O": "ABCDEF",
        "P": "ABEFG",
        "Q": "ABCFG",      # not great
        "R": "AEFGBC",     # not great (P + right)
        "S": "AFGCD",
        "T": "FGED",       # not great
        "U": "BCDEF",
        "V": "BCDEF",      # same as U
        "W": "BCDEF",      # same as U
        "X": "FGEBC",      # same as H
        "Y": "FGBCD",      # not great
        "Z": "ABGED",      # like 2
        "-": "G",
        " ": "",
    }

    seglist = codes.get(chU, "")  # unknown -> blank
    strokes = []
    for s in seglist:
        strokes.append(segs[s])
    return strokes



def plot_plan_xy(plan, title="Simulated pen-tip XY path (pen-down only)", color="k"):
    poses = plan["poses"]
    if len(poses) == 0:
        print("No poses to plot.")
        return

    XY = np.array([T[:2, 3] for T in poses], dtype=float)
    pen = np.array(plan.get("pen_down", [True]*len(poses)), dtype=bool)

    plt.figure()
    i = 0
    while i < len(XY):
        if not pen[i]:
            i += 1
            continue
        j = i
        while j < len(XY) and pen[j]:
            j += 1
        # Force SAME color every run (no color cycling)
        plt.plot(XY[i:j, 0], XY[i:j, 1], color=color)
        i = j

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(title)
    plt.show()

def T_trans(x, y, z):
    T = np.eye(4, dtype=float)
    T[:3, 3] = [x, y, z]
    return T

def make_T(p, R=None):
    """Homogeneous transform from position p (3,) and rotation R (3,3)."""
    T = np.eye(4)
    T[:3, 3] = np.asarray(p, dtype=float).reshape(3,)
    T[:3, :3] = np.eye(3) if R is None else np.asarray(R, dtype=float).reshape(3, 3)
    return T


def test_initial_pose_then_draw(
    Tbt_init,
    *,
    Tboard,
    T6t,
    s="A",
    robot=None,
    dry_run=True,
    z0=0.10,
    z_touch=0.0,
    speed=0.5,
    accel=0.5,
):
    """
    Integration test:

    1) Move robot to user-specified Tbt_init
    2) Verify FK matches Tbt_init numerically
    3) Call draw_string starting from that configuration
    """

    print("\n=== USER-SPECIFIED INITIAL POSE TEST ===")
    print("Tbt_init position:", Tbt_init[:3, 3])

    # -------------------------
    # Step 1: Move to initial pose
    # -------------------------
    out = set_initial_pose(
        Tbt_init,
        T6t=T6t,
        robot=robot,
        simulate=False,
        dry_run=dry_run,
        movej_kwargs=dict(speed=speed, accel=accel),
        on_fail="raise",
    )

    theta_mod0 = out["theta_mod"]
    q_class0   = out["q_class"]

    # -------------------------
    # Step 2: Numerical verification
    # -------------------------
    Tcheck = fk(q_class0, T6t)
    pos_err = np.linalg.norm(Tcheck[:3, 3] - Tbt_init[:3, 3])

    R_err = Tcheck[:3, :3].T @ Tbt_init[:3, :3]
    rot_err = np.linalg.norm(so3_log(R_err))

    print("\n=== FK Verification ===")
    print("Position error (m):", pos_err)
    print("Rotation error (rad):", rot_err)

    if pos_err > 1e-4 or rot_err > 1e-4:
        print("WARNING: Initial pose mismatch exceeds tolerance.")

    # -------------------------
    # Step 3: Draw string from same state
    # -------------------------
    print("\n=== Drawing string from that initial configuration ===")

    plan = draw_string(
        s,
        Tboard=Tboard,
        T6t=T6t,
        z0=z0,
        z_touch=z_touch,
        robot=robot,
        simulate=False,
        dry_run=dry_run,
        thetaSeed0=theta_mod0,
        move_to_first_pose=False,   # critical
        movej_kwargs=dict(speed=speed, accel=accel),
        on_fail="raise",
    )

    return out, plan


if __name__ == "__main__":
    try:
        from ta_utils import ExternalSafetyFilter
        external_filter = ExternalSafetyFilter
        print("ExternalSafetyFilter imported.")
    except Exception:
        external_filter = None
        print("ExternalSafetyFilter not available (simulation only).")

    # connect to UR10e
    # UR_IP = "192.168.0.2"
    # robot = urx.Robot(UR_IP)

    # specify max acc and vel
    aj, vj = 1.0, 0.5   # rad/s^2, rad/s (from your Lab 1 file)
    al, vl = 0.1, 0.03  # m/s^2, m/s

    # IMPORTANT: decide tool-offset convention:
    # robot.set_tcp((0,0,0,0,0,0)) # may need to set this
    robot = 1






    # # -------------------------
    # # Part 1 in lab
    # # -------------------------

    # # plot word from pickle
    # plot_or_execute_word("JointAnglesPractice.pickle", fk, show=True, execute=False)

    # # dry-run execute robot movements
    # plot_or_execute_word("JointAnglesPractice.pickle", fk, execute=True, dry_run=True, external_safety_filter=safety_check) # or external_filter

    # # REAL ROBOT MOVEMENTS
    # plot_or_execute_word(
    #     "JointAnglesPractice.pickle",
    #     fk,
    #     execute=True,
    #     robot=robot,
    #     movej_fn=movej_fn,
    #     movej_kwargs=dict(aj=1.0, vj=0.5, wait=True),
    #     external_safety_filter=safety_check, 
    # )

    # robot.close()

    # # read in yaml file
    # # [fill] add code to execute commands based on yaml file read in









    # # -------------------------
    # # Part 2 in lab
    # # -------------------------

    # ### Part 0 ###

    # # Show that full ik pipeline works with respect to kinematics
    # validate_ik_pipeline(N=10, fk_func=fk, safety_check_func=safety_check)




    # ### Part a ###

    # # specify pure translation of tool frame (IN FRAME 6 COORDINATES)
    # dx, dy, dz = 0, 0, 0  # meters
    # T6t = T_trans(dx, dy, dz)

    # # # or specify full transformation matrix (IN FRAME 6 COORDINATES)
    # # T6t = np.eye(4, dtype=float)
    # # T6t[:3, 3] = [0, 0, 0] # can add rotation 

    # print("Passed part a")




    # ### Part b/e/f ###

    # # Set hover and touch heights
    # z0 = .1 # adjust this to raise the hover height
    # z_touch = .05 # adjust this to raise the touch height

    # # User-specified initial pose in base frame (IN WORLD/BASE COORDINATES)
    # Tbt_init = np.eye(4)
    # Tbt_init[:3, :3] = _tool_orientation_not_axis_aligned()
    # Tbt_init[:3, 3]  = [0.40, -0.10, 0.10]  # ADJUST z0 HERE
 
    # # Move to initial pose (IK + FK verification, but no robot)
    # init_out = set_initial_pose(
    #     Tbt_init,
    #     T6t=T6t,
    #     thetaSeed=None,
    #     fk_func=fk,
    #     safety_check_func=safety_check,
    #     external_safety_filter=external_filter,
    #     simulate=False,                
    #     dry_run=True,                  # switch to False for REAL RUN on robot
    #     robot=robot,
    #     movej_fn=movej_fn,
    #     verify_fk=True,
    #     on_fail="raise",
    # )

    # print("\n=== set_initial_pose dry_run ===")
    # print("success:", init_out["success"])
    # print("executed:", init_out.get("executed"))
    # print("q_class:", np.array2string(init_out["q_class"], precision=4))
    # print("verify:", init_out.get("verify", {}))

    # thetaSeed0 = init_out["theta_mod"]

    # # end of Move to inital pose ---------------------------


    # # draw_string starting from that initial condition 

    # # Put board origin in reachable region (SET THIS) (IN WORLD/BASE COORDINATES TbWb)
    # # If board is not placed yet just set pseudo position relavant to movement.
    # Tboard = np.eye(4) 
    # Tboard[:3, 3] = [0.40, -0.10, 0.0]

    # # calculate difference between starting location and location of first letter on Tboard
    # Tbt_first = np.eye(4)
    # Tbt_first[:3,:3] = _tool_orientation_not_axis_aligned()
    # Tbt_first[:3,3]  = [Tboard[0,3] + 0.0, Tboard[1,3] + 0.0, .1]  
    # print("init - first (m):", np.linalg.norm(Tbt_init[:3,3] - Tbt_first[:3,3]))

    # out_str = draw_string(
    #     "lets get this bread",
    #     Tboard=Tboard,
    #     T6t=T6t,
    #     z0=z0,
    #     z_touch=z_touch,
    #     n_per_segment=6,
    #     fk_func=fk,
    #     safety_check_func=safety_check,
    #     external_safety_filter=external_filter,
    #     simulate=False,
    #     dry_run=True,                  # switch to False for REAL RUN on robot
    #     robot=robot,
    #     movej_fn=movej_fn,
    #     translate_fn=translate_fn,
    #     thetaSeed0=thetaSeed0,
    #     move_to_first_pose=False,      # CRITICAL: "take off" from the initial pose
    #     print_p_end=True,
    # )

    # print("\n=== draw_string dry_run ===")
    # print("success:", out_str["success"])
    # print("p_end_final:", out_str.get("p_end_final"))

    # # end of draw string from initial position -------------------------------


    # # Plot the planned XY trajectory of out_str (pen-down only)
    # poses_concat = []
    # pen_concat = []
    # for ch_out in out_str["characters"]:
    #     poses_concat.extend(ch_out["poses"])
    #     pen_concat.extend(ch_out["pen_down"])

    # plan_xy = {"poses": poses_concat, "pen_down": pen_concat}
    # plot_plan_xy(plan_xy, "DRY-RUN integration: set_initial_pose -> draw_string()")
    # plt.show()

    # # end of plotting the planned XY trajectory of out_str (pen-down only) ---------

    # print("Passed part b")




    # ### Part c ###

    # # Measure and set board position (SET THIS) (IN WORLD/BASE COORDINATES TbWb)
    # Tboard = np.eye(4) 
    # Tboard[:3, 3] = [0.40, -0.10, 0.0]

    # # RESET pure translation of tool frame (IN FRAME 6 COORDINATES)
    # dx, dy, dz = 0, 0, 0  # meters
    # T6t = T_trans(dx, dy, dz)

    # # # or specify full transformation matrix (IN FRAME 6 COORDINATES)
    # # T6t = np.eye(4, dtype=float)
    # # T6t[:3, 3] = [0, 0, 0] # can add rotation 

    # print("Passed part c")






    

#    plan = draw_string(
#        "Lets get this bread",
#        Tboard=Tboard, # 4x4 homogeneous transform of the board frame in WORLD frame and defines where the lower-left of the text grid lives in space.
#        T6t=T6t, # 4x4 transform from tool frame to end-effector (frame 6 to tool frame)
#        z0=0.12, # Pen-up height (meters) in board frame.
#        z_touch=0.08, # Pen-down height (meters) in board frame.
#        char_w=0.03, # Character width (meters) in board X direction.
#        char_h=0.06, # Character height (meters) in board Y direction.
#        char_spacing=0.01, # Horizontal spacing between characters (meters).
#        max_width=1, # Maximum total text width (meters) before right boundary of board frame.
#        fk_func=fk, # Forward kinematics function.
#        safety_check_func=safety_check, # Internal safety validation (does not let go below z=0)
#        external_safety_filter=None, # checks for external safety filter code
#        simulate=False,       # True: nothing is executed, only plan is created. False: Actually execute plan.
#        dry_run=True,         # Only relevant if simulate=False. True: Execute but prints commands to console. (for debugging) False: robot execute commands. (real robot mode) 
#        print_p_end=True,     # prints p_end each step
#        on_fail="raise", # "raise": throw exception on IK/safety failure. "skip": skip infeasible character/pose and continue.
#    )
#
#    print("Planned poses:", len(plan["poses"]), "scale:", plan.get("scale", 1.0))
#    plot_plan_xy(plan, color='k')



    