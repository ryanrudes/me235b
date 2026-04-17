"""UR10e kinematics: FK, closed-form IK, and round-trip validation.

Everything the kinematics layer needs to know lives on the `UR10e` class:
DH constants, the tool offset (`T6t`), joint limits, and optional safety hooks.
Callers just instantiate a `UR10e` once and call methods on it; they don't
have to thread `fk_func`, `safety_check_func`, `joint_limits_rad`, etc. through
every call site.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .transforms import as_6vec, inv_T, wrap_to_pi


def dh_classical(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Classical DH transform: T = Rz(theta) * Tz(d) * Tx(a) * Rx(alpha)."""
    ct, st = float(np.cos(theta)), float(np.sin(theta))
    ca, sa = float(np.cos(alpha)), float(np.sin(alpha))
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def dh_modified(a_im1: float, alpha_im1: float, d_i: float, theta_i: float) -> np.ndarray:
    """Craig modified DH: T = Rx(alpha_{i-1}) * Tx(a_{i-1}) * Rz(theta_i) * Tz(d_i)."""
    ca, sa = float(np.cos(alpha_im1)), float(np.sin(alpha_im1))
    ct, st = float(np.cos(theta_i)), float(np.sin(theta_i))
    return np.array(
        [
            [ct, -st, 0.0, a_im1],
            [st * ca, ct * ca, -sa, -d_i * sa],
            [st * sa, ct * sa, ca, d_i * ca],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _is_duplicate(q: np.ndarray, Q: np.ndarray, tol: float = 1e-3) -> bool:
    q = as_6vec(q)
    if Q.size == 0:
        return False
    for k in range(Q.shape[0]):
        if np.linalg.norm(wrap_to_pi(q - Q[k, :])) < tol:
            return True
    return False


class UR10e:
    """UR10e kinematics: forward, inverse, safety, and validation.

    Parameters
    ----------
    T6t : (4,4) array-like, optional
        Frame6 to tool transform. Defaults to identity.
    joint_limits_rad : list of (lo, hi), optional
        Per-joint limits in radians. Defaults to (-2 pi, 2 pi) on every joint.
    joint_limit_margin_rad : float
        Extra margin subtracted from each joint's usable range during IK.
    external_safety_filter : callable, optional
        Predicate on classical joint angles (``q_class``). If provided, IK
        candidates must satisfy it in addition to ``safety_check``.
    """

    # Classical DH parameters.
    a = np.array([0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0], dtype=float)
    d = np.array([0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655], dtype=float)
    alpha = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0], dtype=float)

    MOD_TO_CLASSICAL_OFFSET = np.array(
        [0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0],
        dtype=float,
    )

    def __init__(
        self,
        T6t: np.ndarray | None = None,
        *,
        joint_limits_rad: Sequence[tuple[float, float]] | None = None,
        joint_limit_margin_rad: float = 0.0,
        external_safety_filter: Callable[[np.ndarray], bool] | None = None,
    ) -> None:
        self.T6t = np.eye(4, dtype=float) if T6t is None else np.asarray(T6t, dtype=float)
        if self.T6t.shape != (4, 4):
            raise ValueError(f"T6t must be 4x4; got {self.T6t.shape}.")

        if joint_limits_rad is None:
            joint_limits_rad = [(-2.0 * np.pi, 2.0 * np.pi)] * 6
        if len(joint_limits_rad) != 6:
            raise ValueError("joint_limits_rad must be length-6 list of (lo, hi) tuples.")
        self.joint_limits_rad = [(float(lo), float(hi)) for lo, hi in joint_limits_rad]
        self.joint_limit_margin_rad = float(joint_limit_margin_rad)
        self.external_safety_filter = external_safety_filter

    def fk(self, q, *, include_tool: bool = True) -> np.ndarray:
        """Forward kinematics in classical DH.

        Returns the base->tool (or base->frame6) transform depending on ``include_tool``.
        """
        q = as_6vec(q)
        T_B6 = np.eye(4, dtype=float)
        for i in range(6):
            T_B6 = T_B6 @ dh_classical(self.a[i], self.alpha[i], self.d[i], q[i])
        return T_B6 @ self.T6t if include_tool else T_B6

    def safety_check(self, q) -> bool:
        """Return True iff no frame origin (or the tool point) drops below z=0."""
        q = as_6vec(q)
        T = np.eye(4, dtype=float)
        for i in range(6):
            T = T @ dh_classical(self.a[i], self.alpha[i], self.d[i], q[i])
            if T[2, 3] < 0.0:
                return False
        T_tool = T @ self.T6t
        return bool(T_tool[2, 3] >= 0.0)

    def dh_modified_to_classical(self, theta_mod) -> np.ndarray:
        """Convert Williams/modified-DH angles to classical UR joint angles."""
        theta_mod = as_6vec(theta_mod)
        return wrap_to_pi(theta_mod + self.MOD_TO_CLASSICAL_OFFSET)

    def classical_to_dh_modified(self, q_class) -> np.ndarray:
        """Inverse of :meth:`dh_modified_to_classical`; used inside the IK solver."""
        q_class = as_6vec(q_class)
        return wrap_to_pi(q_class - self.MOD_TO_CLASSICAL_OFFSET)

    def _within_limits(self, q: np.ndarray) -> bool:
        margin = self.joint_limit_margin_rad
        for i in range(6):
            lo, hi = self.joint_limits_rad[i]
            if not (lo <= q[i] <= hi):
                return False
            if margin > 0 and ((q[i] < lo + margin) or (q[i] > hi - margin)):
                return False
        return True

    def ik(self, T_bt, *, theta_seed=None) -> tuple[np.ndarray, np.ndarray, dict]:
        """Analytical UR10e IK.

        Returns ``(theta_mod, theta_all, info)``. ``theta_mod`` is the candidate
        (in Williams/modified-DH variables) closest to ``theta_seed`` under the
        wrapped-distance metric. ``theta_all`` stacks every surviving candidate.
        ``info`` carries metadata (see code).
        """
        T_bt = np.asarray(T_bt, dtype=float)
        if T_bt.shape != (4, 4):
            raise ValueError("ik: T_bt must be 4x4.")

        theta_seed = np.zeros(6, dtype=float) if theta_seed is None else as_6vec(theta_seed)

        a, d, alpha = self.a, self.d, self.alpha

        T_B6_des = T_bt @ inv_T(self.T6t)
        R06 = T_B6_des[:3, :3]
        p06 = T_B6_des[:3, 3]

        def _fail(msg: str, solutions: list) -> tuple[np.ndarray, np.ndarray, dict]:
            return (
                np.full(6, np.nan, dtype=float),
                np.zeros((0, 6), dtype=float),
                {
                    "success": False,
                    "message": msg,
                    "thetaAll": np.zeros((0, 6), dtype=float),
                    "qAll_classical": np.zeros((0, 6), dtype=float),
                    "solutions": solutions,
                },
            )

        # q1: wrist center of frame 5 projected to base XY plane.
        p05 = p06 - d[5] * R06[:, 2]
        px, py = float(p05[0]), float(p05[1])
        r_xy = float(np.hypot(px, py))
        if r_xy < abs(d[3]) - 1e-12:
            return _fail("ik: no solution (wrist center too close to base axis for q1).", [])

        q_sols: list[np.ndarray] = []
        sol_details: list[dict] = []

        phi = float(np.arctan2(py, px))
        delta = float(np.arccos(float(np.clip(d[3] / r_xy, -1.0, 1.0))))
        q1_candidates = [phi + delta + np.pi / 2, phi - delta + np.pi / 2]

        for q1 in q1_candidates:
            q1 = float(wrap_to_pi(q1))

            arg5 = float(np.clip((p06[0] * np.sin(q1) - p06[1] * np.cos(q1) - d[3]) / d[5], -1.0, 1.0))
            q5_base = float(np.arccos(arg5))

            for q5 in (q5_base, -q5_base):
                q5 = float(wrap_to_pi(q5))
                s5 = float(np.sin(q5))
                if abs(s5) < 1e-8:
                    continue

                q6 = float(
                    np.arctan2(
                        (-R06[0, 1] * np.sin(q1) + R06[1, 1] * np.cos(q1)) / s5,
                        (R06[0, 0] * np.sin(q1) - R06[1, 0] * np.cos(q1)) / s5,
                    )
                )
                q6 = float(wrap_to_pi(q6))

                T01 = dh_classical(a[0], alpha[0], d[0], q1)
                T56 = dh_classical(a[5], alpha[5], d[5], q6)
                T45 = dh_classical(a[4], alpha[4], d[4], q5)
                T14 = inv_T(T01) @ T_B6_des @ inv_T(T56) @ inv_T(T45)

                x, y = float(T14[0, 3]), float(T14[1, 3])
                r2 = x * x + y * y

                cos3 = (r2 - a[1] ** 2 - a[2] ** 2) / (2.0 * a[1] * a[2])
                if cos3 < -1.0 - 1e-8 or cos3 > 1.0 + 1e-8:
                    continue
                cos3 = float(np.clip(cos3, -1.0, 1.0))
                q3_base = float(np.arccos(cos3))

                for q3 in (q3_base, -q3_base):
                    q3 = float(wrap_to_pi(q3))
                    q2 = float(
                        np.arctan2(y, x)
                        - np.arctan2(a[2] * np.sin(q3), a[1] + a[2] * np.cos(q3))
                    )
                    q2 = float(wrap_to_pi(q2))

                    q234 = float(np.arctan2(T14[1, 0], T14[0, 0]))
                    q4 = float(wrap_to_pi(q234 - q2 - q3))

                    q_sol = wrap_to_pi(np.array([q1, q2, q3, q4, q5, q6], dtype=float))

                    if not self._within_limits(q_sol):
                        continue
                    if not self.safety_check(q_sol):
                        continue
                    if self.external_safety_filter is not None and not self.external_safety_filter(q_sol):
                        continue

                    stacked = np.vstack(q_sols) if q_sols else np.zeros((0, 6), dtype=float)
                    if _is_duplicate(q_sol, stacked, tol=1e-3):
                        continue

                    q_sols.append(q_sol)
                    sol_details.append({"branch": "analytic", "q": q_sol.copy()})

        if not q_sols:
            return _fail(
                "ik: no solution found (analytic branches invalid / rejected by guardrails).",
                sol_details,
            )

        q_all = np.vstack(q_sols)
        theta_all = np.vstack([self.classical_to_dh_modified(q_all[k, :]) for k in range(q_all.shape[0])])

        dists = np.array(
            [np.linalg.norm(wrap_to_pi(theta_all[k, :] - theta_seed)) for k in range(theta_all.shape[0])],
            dtype=float,
        )
        idx = int(np.argmin(dists))

        info = {
            "success": True,
            "message": f"ik: analytic found {theta_all.shape[0]} solution(s).",
            "thetaAll": theta_all,
            "qAll_classical": q_all,
            "solutions": sol_details,
            "chosen_index": idx,
            "joint_limits_rad": self.joint_limits_rad,
            "joint_limit_margin_rad": self.joint_limit_margin_rad,
        }
        return theta_all[idx, :].copy(), theta_all, info

    def validate_pipeline(self, n: int = 25, rng_seed: int = 0) -> dict:
        """FK -> IK -> FK round-trip sanity check, reporting mean/max errors."""
        rng = np.random.default_rng(rng_seed)

        pos_errs: list[float] = []
        rot_errs: list[float] = []
        fails = 0

        for _ in range(n):
            theta_true = (rng.random(6) - 0.5) * (2.0 * np.pi / 3.0)
            q_true = self.dh_modified_to_classical(theta_true)
            T_bt = self.fk(q_true)

            theta_ik, _, info = self.ik(T_bt, theta_seed=np.zeros(6))
            if (not info.get("success", False)) or np.any(np.isnan(theta_ik)):
                fails += 1
                continue

            q_hat = self.dh_modified_to_classical(theta_ik)
            T_hat = self.fk(q_hat)

            pos_errs.append(float(np.linalg.norm(T_bt[:3, 3] - T_hat[:3, 3])))
            R = T_bt[:3, :3].T @ T_hat[:3, :3]
            c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
            rot_errs.append(float(np.arccos(c)))

        summary = {
            "n": n,
            "failures": fails,
            "pos_err_mean": float(np.mean(pos_errs)) if pos_errs else float("nan"),
            "pos_err_max": float(np.max(pos_errs)) if pos_errs else float("nan"),
            "rot_err_mean": float(np.mean(rot_errs)) if rot_errs else float("nan"),
            "rot_err_max": float(np.max(rot_errs)) if rot_errs else float("nan"),
        }

        print(f"\nRound-trip validation over N={n} tests:")
        print(f"  Position error (m):   mean {summary['pos_err_mean']:.3e}, max {summary['pos_err_max']:.3e}")
        print(f"  Rotation error (rad): mean {summary['rot_err_mean']:.3e}, max {summary['rot_err_max']:.3e}")
        print(f"  Failures: {fails} / {n}")

        return summary
