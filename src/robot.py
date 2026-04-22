"""Thin, class-based wrapper around a UR10e / ``urx`` connection.

``RobotController`` owns:

- the ``UR10e`` kinematics (FK, IK, safety check),
- speed/acceleration defaults (``aj``, ``vj``, ``al``, ``vl``),
- the last IK seed (``theta_mod``) plus the current classical joint vector,
- the simulate / dry-run flags, and
- an optional :class:`me235b.sim.SimulationRenderer` that observes every
  joint target / translate / gripper event.

Callers never need to pass ``fk_func``, ``safety_check_func``, ``thetaSeed0``,
``simulate``, ``dry_run``, ``robot``, ``movej_fn`` or ``movej_kwargs`` around.
They construct a controller once and call methods on it.

``urx`` is imported lazily inside :meth:`RobotController.connect` so the rest
of the package (including the CLI and unit tests) can run without ``urx``
installed.
"""

from __future__ import annotations

from typing import Any

import pyrobotiqr as rq
import numpy as np

from kinematics import UR10e
from sim import NullRenderer, SimulationRenderer
from transforms import shortest_joint_motion_target, so3_log


class RobotController:
    def __init__(
        self,
        kinematics: UR10e,
        *,
        aj: float = 1.0,
        vj: float = 0.5,
        al: float = 0.1,
        vl: float = 0.03,
        simulate: bool = True,
        dry_run: bool = False,
        verbose: bool = True,
        renderer: SimulationRenderer | None = None,
        step_duration: float = 0.4,
    ) -> None:
        self.kin = kinematics
        self.aj, self.vj = float(aj), float(vj)
        self.al, self.vl = float(al), float(vl)
        self.simulate = bool(simulate)
        self.dry_run = bool(dry_run)
        self.verbose = bool(verbose)
        self.renderer: SimulationRenderer = renderer if renderer is not None else NullRenderer()
        self.step_duration = float(step_duration)

        self.robot: Any | None = None
        self.theta_seed: np.ndarray = np.zeros(6, dtype=float)
        self.current_q_class: np.ndarray = np.zeros(6, dtype=float)

    def connect(self, ip: str, *, tcp: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)) -> None:
        """Open a ``urx.Robot`` connection. Only needed for real hardware."""
        try:
            import urx
        except ImportError as exc:
            raise RuntimeError(
                "RobotController.connect requires the 'urx' package, which is not installed. "
                "Install it (e.g. `uv add urx`) to talk to real hardware, or keep simulate=True."
            ) from exc
        self.robot = urx.Robot(ip)
        if tcp is not None:
            self.robot.set_tcp(tuple(tcp))
        self.gripper = rq.RobotiqGripper(tcp_host="192.168.0.2", port=54321)
        self.gripper.connect()
        self.gripper.activate()
        self.gripper.start()

    def close(self) -> None:
        if self.robot is not None:
            try:
                self.robot.close()
            finally:
                self.robot = None

    def __enter__(self) -> "RobotController":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()

    @property
    def is_live(self) -> bool:
        """True iff we will dispatch commands to a real robot on the next call."""
        return (not self.simulate) and (not self.dry_run)

    def _dispatch_movej(self, q_class: np.ndarray, *, wait: bool) -> None:
        if self.robot is None:
            raise RuntimeError(
                "RobotController.movej requires an active connection; call connect(ip) first "
                "or use simulate=True / dry_run=True."
            )
        self.robot.movej(tuple(q_class.tolist()), self.aj, self.vj, wait=wait)
        self.robot.stopj(self.aj)

    def _dispatch_translate(self, dx: float, dy: float, dz: float, *, wait: bool) -> None:
        if self.robot is None:
            raise RuntimeError("RobotController.translate requires an active connection.")
        self.robot.translate((float(dx), float(dy), float(dz)), self.al, self.vl, wait=wait)
        self.robot.stopl(self.al)

    def current_tool_pose(self) -> np.ndarray:
        """Forward kinematics of the last committed joint state."""
        return self.kin.fk(self.current_q_class)

    def movej(self, q_class, *, wait: bool = True) -> np.ndarray:
        """Safety-filter and dispatch a ``movej``. Returns the issued classical joints."""
        q_class = np.asarray(q_class, dtype=float).reshape(6)
        q_cmd = shortest_joint_motion_target(self.current_q_class, q_class)
        self.renderer.on_demo_checkpoint()
        if self.kin.external_safety_filter is not None and not self.kin.external_safety_filter(q_cmd):
            raise RuntimeError("movej: rejected by external safety filter.")

        if self.simulate:
            if self.verbose:
                print(f"[robot:sim] movej: {np.array2string(q_cmd, precision=4)}")
        elif self.dry_run:
            if self.verbose:
                print(f"[robot:dry] movej: {np.array2string(q_cmd, precision=4)}")
        else:
            self._dispatch_movej(q_cmd, wait=wait)

        self.renderer.on_joint_step(q_cmd, duration=self.step_duration)
        self.current_q_class = q_cmd.copy()
        return q_cmd

    def translate(self, dx: float, dy: float, dz: float, *, wait: bool = True) -> tuple[float, float, float]:
        """Cartesian translate. In simulate / dry-run mode the move is emulated via IK."""
        dx_f, dy_f, dz_f = float(dx), float(dy), float(dz)

        if self.simulate:
            if self.verbose:
                print(f"[robot:sim] translate: dx={dx_f:+.4f} dy={dy_f:+.4f} dz={dz_f:+.4f}")
            self._emulate_translate(dx_f, dy_f, dz_f)
            return dx_f, dy_f, dz_f
        if self.dry_run:
            if self.verbose:
                print(f"[robot:dry] translate: dx={dx_f:+.4f} dy={dy_f:+.4f} dz={dz_f:+.4f}")
            self._emulate_translate(dx_f, dy_f, dz_f)
            return dx_f, dy_f, dz_f

        self._dispatch_translate(dx_f, dy_f, dz_f, wait=wait)
        return dx_f, dy_f, dz_f

    def _emulate_translate(self, dx: float, dy: float, dz: float) -> None:
        """Solve IK for the new tool pose after a Cartesian delta and publish it."""
        T_current = self.current_tool_pose()
        T_target = T_current.copy()
        T_target[:3, 3] += np.array([dx, dy, dz], dtype=float)

        theta_mod, _, info = self.kin.ik(T_target, theta_seed=self.theta_seed)
        if (not info.get("success", False)) or np.any(np.isnan(theta_mod)):
            if self.verbose:
                print(f"[robot:sim] translate IK failed ({info.get('message','')}); renderer frozen.")
            return

        q_new = self.kin.dh_modified_to_classical(theta_mod)
        q_cmd = shortest_joint_motion_target(self.current_q_class, q_new)
        self.renderer.on_demo_checkpoint()
        self.renderer.on_joint_step(q_cmd, duration=self.step_duration)
        self.current_q_class = q_cmd.copy()
        self.theta_seed = theta_mod.copy()

    def gripper_open(self) -> None:
        if self.is_live:
            self.gripper.open()
        elif self.verbose:
            print("[robot] gripper_open (no-op; simulate/dry_run)")
        self.renderer.on_gripper("open")

    def gripper_close(self, *, grasped_width_m: float | None = None) -> None:
        """Close the gripper.

        ``grasped_width_m`` (optional) is forwarded to the attached renderer so
        adaptive grippers can stop their fingers at the block's width instead
        of mashing them fully together. It's ignored for live hardware control.
        """
        if self.is_live:
            self.gripper.close()
        elif self.verbose:
            print("[robot] gripper_close (no-op; simulate/dry_run)")
        self.renderer.on_gripper("closed", grasped_width_m=grasped_width_m)

    def move_to_pose(
        self,
        T_target: np.ndarray,
        *,
        kinematics: UR10e | None = None,
        theta_seed: np.ndarray | None = None,
        wait: bool = True,
    ) -> tuple[bool, np.ndarray, dict]:
        """IK -> safety filter -> movej. Updates the internal IK seed on success.

        ``kinematics`` lets a caller run IK against a different tool offset (e.g. a
        camera rigged to frame 5) without mutating ``self.kin``.
        """
        kin = kinematics if kinematics is not None else self.kin
        seed = self.theta_seed if theta_seed is None else np.asarray(theta_seed, dtype=float).reshape(6)

        theta_mod, _, info = kin.ik(T_target, theta_seed=seed)
        if (not info.get("success", False)) or np.any(np.isnan(theta_mod)):
            if self.verbose:
                print(f"[robot] move_to_pose: IK failed ({info.get('message','')})")
            return False, seed, info

        q_class = kin.dh_modified_to_classical(theta_mod)
        self.movej(q_class, wait=wait)
        self.theta_seed = theta_mod.copy()
        return True, theta_mod, info

    def move_to_camera_pose(
        self,
        T_target: np.ndarray,
        T5c: np.ndarray,
        *,
        theta_seed: np.ndarray | None = None,
        wait: bool = True,
        position_only: bool = False,
        trans_weight: float = 1.0,
        rot_weight: float = 1.0,
        pos_tol: float = 1e-4,
        rot_tol: float = 1e-3,
        max_iter: int = 80,
        pre_solve_position: bool = False,
    ) -> tuple[bool, np.ndarray, dict]:
        """IK so ``kin.fk_to_frame(q,5) @ T5c`` matches ``T_target``, then movej.

        Use this for scan poses where the camera is mounted on link 5. The
        standard :meth:`move_to_pose` / flange ``T6t`` chain does not match that
        mount.

        Set ``position_only=True`` when the target is an over-specified full SE(3)
        and you only care about camera **position**. Otherwise use ``trans_weight``
        / ``rot_weight`` to trade a small position error for orientation (e.g. look
        downward at scan points).
        """
        seed = self.theta_seed if theta_seed is None else np.asarray(theta_seed, dtype=float).reshape(6)

        theta_mod, _, info = self.kin.ik_camera_mount(
            T_target,
            T5c,
            theta_seed=seed,
            position_only=position_only,
            trans_weight=trans_weight,
            rot_weight=rot_weight,
            pos_tol=pos_tol,
            rot_tol=rot_tol,
            max_iter=max_iter,
            pre_solve_position=pre_solve_position,
        )
        if (not info.get("success", False)) or np.any(np.isnan(theta_mod)):
            if self.verbose:
                print(f"[robot] move_to_camera_pose: IK failed ({info.get('message','')})")
            return False, seed, info

        q_class = self.kin.dh_modified_to_classical(theta_mod)
        q_cmd = shortest_joint_motion_target(self.current_q_class, q_class)
        self.renderer.on_demo_checkpoint()
        self.movej(q_cmd, wait=wait)
        self.theta_seed = theta_mod.copy()
        return True, theta_mod, info

    def home(
        self,
        T_home: np.ndarray,
        *,
        theta_seed: np.ndarray | None = None,
        verify_fk: bool = True,
        pos_tol: float = 1e-4,
        rot_tol: float = 1e-4,
        wait: bool = True,
    ) -> dict:
        """Drive the arm to ``T_home`` and (optionally) verify the FK result."""
        ok, theta_mod, info = self.move_to_pose(T_target=T_home, theta_seed=theta_seed, wait=wait)
        out: dict = {"success": ok, "info": info, "theta_mod": theta_mod}
        if not ok:
            return out

        q_class = self.kin.dh_modified_to_classical(theta_mod)
        out["q_class"] = q_class

        if verify_fk:
            T_check = self.kin.fk(q_class)
            pos_err = float(np.linalg.norm(T_check[:3, 3] - T_home[:3, 3]))
            R_err = T_check[:3, :3].T @ T_home[:3, :3]
            rot_err = float(np.linalg.norm(so3_log(R_err)))
            out["verify"] = {
                "pos_err_m": pos_err,
                "rot_err_rad": rot_err,
                "pos_tol_m": float(pos_tol),
                "rot_tol_rad": float(rot_tol),
                "ok": (pos_err <= pos_tol and rot_err <= rot_tol),
            }
            if self.verbose:
                print(
                    f"[robot] home FK verify: pos_err={pos_err:.3e} m, "
                    f"rot_err={rot_err:.3e} rad (ok={out['verify']['ok']})"
                )
        return out
