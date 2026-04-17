"""7-segment glyph planner + whiteboard executor.

- :class:`SevenSegFont` is a pure-geometry lookup from characters to 2D strokes.
- :class:`StringDrawer` owns a :class:`RobotController` and drives it to plan
  or draw a character / string.
- :func:`plot_plan_xy`, :func:`plot_pickle_word`, :func:`execute_pickle_word`
  are small free helpers for Lab-2 style plotting and pickle playback.
- :func:`run_draw` is a Typer-friendly CLI entry point that ties it all
  together, optionally driving a :class:`me235b.sim.ViserRenderer`.
"""

from __future__ import annotations

import pickle
from enum import Enum
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import typer
from typing_extensions import Annotated

from .kinematics import UR10e
from .robot import RobotController
from .sim import SimulationRenderer
from .transforms import as_6vec, make_T, tool_orientation_tilted


Stroke = tuple[tuple[float, float], tuple[float, float]]


class SevenSegFont:
    """Characters rendered as 7-segment polyline strokes in local character coordinates."""

    # Canonical 7-seg codes. Unknown / unmapped characters render as blank.
    CODES: dict[str, str] = {
        "0": "ABCDEF",
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
        "B": "FGCDE",
        "C": "ADEF",
        "D": "BCDEG",
        "E": "AFGED",
        "F": "AFGE",
        "G": "AFECD",
        "H": "FGEBC",
        "I": "BC",
        "J": "BCD",
        "K": "FGEBC",
        "L": "DEF",
        "M": "ABCEFG",
        "N": "FGEBC",
        "O": "ABCDEF",
        "P": "ABEFG",
        "Q": "ABCFG",
        "R": "AEFGBC",
        "S": "AFGCD",
        "T": "FGED",
        "U": "BCDEF",
        "V": "BCDEF",
        "W": "BCDEF",
        "X": "FGEBC",
        "Y": "FGBCD",
        "Z": "ABGED",
        "-": "G",
        " ": "",
    }

    def __init__(self, w: float = 0.03, h: float = 0.06) -> None:
        self.w = float(w)
        self.h = float(h)

    def _segments(self) -> dict[str, Stroke]:
        w, h = self.w, self.h
        return {
            "A": ((0, h), (w, h)),
            "B": ((w, h), (w, h / 2)),
            "C": ((w, h / 2), (w, 0)),
            "D": ((0, 0), (w, 0)),
            "E": ((0, h / 2), (0, 0)),
            "F": ((0, h), (0, h / 2)),
            "G": ((0, h / 2), (w, h / 2)),
        }

    def strokes(self, ch: str) -> list[Stroke]:
        if not isinstance(ch, str) or not ch:
            ch = " "
        code = self.CODES.get(ch.upper(), "")
        segs = self._segments()
        return [segs[s] for s in code]


def _pose_from_xy(x: float, y: float, z: float, R: np.ndarray | None = None) -> np.ndarray:
    R = tool_orientation_tilted() if R is None else R
    return make_T([x, y, z], R)


class StringDrawer:
    """Plans and (optionally) executes whiteboard text via a :class:`RobotController`."""

    def __init__(
        self,
        controller: RobotController,
        *,
        z0: float = 0.10,
        z_touch: float = 0.02,
        char_spacing: float = 0.01,
        line_spacing: float = 0.02,
        n_per_segment: int = 8,
        font: SevenSegFont | None = None,
        on_fail: str = "skip",
    ) -> None:
        if on_fail not in ("skip", "raise"):
            raise ValueError(f"on_fail must be 'skip' or 'raise'; got {on_fail!r}.")

        self.controller = controller
        self.kin: UR10e = controller.kin
        self.z0 = float(z0)
        self.z_touch = float(z_touch)
        self.char_spacing = float(char_spacing)
        self.line_spacing = float(line_spacing)
        self.n_per_segment = int(n_per_segment)
        self.font = font if font is not None else SevenSegFont()
        self.on_fail = on_fail

    @property
    def char_w(self) -> float:
        return self.font.w

    @property
    def char_h(self) -> float:
        return self.font.h

    def add_board_visual(
        self,
        T_board: np.ndarray,
        *,
        width: float = 0.40,
        height: float = 0.30,
        color: tuple[int, int, int] = (240, 240, 240),
    ) -> str:
        """Drop a thin whiteboard panel into the renderer at ``T_board``.

        The board is placed so its lower-left corner sits at the board frame origin
        (matching how :meth:`plan_string` expects the cursor to start at (0, 0)).
        """
        T = np.asarray(T_board, dtype=float).copy()
        T[:3, 3] = T[:3, 3] + T[:3, :3] @ np.array([width / 2, height / 2, -0.003], dtype=float)
        return self.controller.renderer.add_box(
            T,
            dimensions=(float(width), float(height), 0.004),
            label="whiteboard",
            color=color,
        )

    def _solve_pose(self, T: np.ndarray, theta_seed: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
        theta_mod, _, info = self.kin.ik(T, theta_seed=theta_seed)
        if (not info.get("success", False)) or np.any(np.isnan(theta_mod)):
            return None, None, info
        q_class = self.kin.dh_modified_to_classical(theta_mod)
        return theta_mod, q_class, info

    def plan_character(
        self,
        T_start: np.ndarray,
        ch: str,
        *,
        theta_seed: np.ndarray | None = None,
    ) -> dict:
        """Plan IK waypoints (hover-down-stroke-up) for a single character.

        Returns a dict with ``poses``, ``theta_mod``, ``q_class``, ``pen_down``,
        ``failures``, ``success`` and bookkeeping fields. No robot commands.
        """
        T_start = np.asarray(T_start, dtype=float)
        if T_start.shape != (4, 4):
            raise ValueError("plan_character: T_start must be 4x4.")

        x0, y0, _ = T_start[:3, 3]
        R = tool_orientation_tilted()
        seed = self.controller.theta_seed if theta_seed is None else as_6vec(theta_seed)

        poses: list[np.ndarray] = []
        q_class_list: list[np.ndarray] = []
        theta_mod_list: list[np.ndarray] = []
        pen_flags: list[bool] = []
        failures: list[dict] = []

        def append(T: np.ndarray, *, pen_down: bool) -> bool:
            nonlocal seed
            th, qc, info = self._solve_pose(T, seed)
            if th is None:
                failures.append({"pose": T.copy(), "info": info})
                if self.on_fail == "raise":
                    raise RuntimeError(f"plan_character: IK failed for {ch!r}: {info.get('message','')}")
                return False
            poses.append(T)
            theta_mod_list.append(th)
            q_class_list.append(qc)
            pen_flags.append(bool(pen_down))
            seed = th.copy()
            return True

        if not append(_pose_from_xy(x0, y0, self.z0, R), pen_down=False):
            return {
                "poses": [],
                "theta_mod": [],
                "q_class": [],
                "pen_down": [],
                "failures": failures,
                "success": False,
                "char": ch,
                "origin_xy": (x0, y0),
            }

        for (sx, sy), (ex, ey) in self.font.strokes(ch):
            if not append(_pose_from_xy(x0 + sx, y0 + sy, self.z0, R), pen_down=False):
                continue
            if not append(_pose_from_xy(x0 + sx, y0 + sy, self.z_touch, R), pen_down=True):
                continue

            for t in np.linspace(0.0, 1.0, self.n_per_segment):
                x = x0 + (1 - t) * sx + t * ex
                y = y0 + (1 - t) * sy + t * ey
                if not append(_pose_from_xy(x, y, self.z_touch, R), pen_down=True):
                    break

            append(_pose_from_xy(x0 + ex, y0 + ey, self.z0, R), pen_down=False)

        return {
            "poses": poses,
            "theta_mod": theta_mod_list,
            "q_class": q_class_list,
            "pen_down": pen_flags,
            "failures": failures,
            "success": not failures,
            "char": ch,
            "origin_xy": (x0, y0),
        }

    def draw_character(
        self,
        T_start: np.ndarray,
        ch: str,
        *,
        theta_seed: np.ndarray | None = None,
    ) -> dict:
        """Plan + drive the robot through a single character.

        Movement strategy: ``movej`` to the first planned pose, then Cartesian
        ``translate`` for each subsequent delta. Execution (or lack thereof) is
        entirely up to :attr:`self.controller` (simulate / dry_run / live). Each
        executed pose is reported to the controller's renderer as a trail point
        so simulated runs can visualize the whiteboard strokes.
        """
        plan = self.plan_character(T_start, ch, theta_seed=theta_seed)
        if not plan["poses"]:
            plan["executed"] = False
            return plan

        renderer = self.controller.renderer
        poses = plan["poses"]
        pen_down = plan["pen_down"]

        self.controller.movej(plan["q_class"][0])
        renderer.trail_point(poses[0][:3, 3], pen_down=pen_down[0])

        if plan["theta_mod"]:
            self.controller.theta_seed = plan["theta_mod"][-1].copy()

        for k in range(1, len(poses)):
            dp = poses[k][:3, 3] - poses[k - 1][:3, 3]
            self.controller.translate(float(dp[0]), float(dp[1]), float(dp[2]))
            renderer.trail_point(poses[k][:3, 3], pen_down=pen_down[k])

        plan["executed"] = True
        return plan

    def _board_xyz_to_base(self, T_board: np.ndarray, x: float, y: float, z: float) -> np.ndarray:
        p = T_board @ np.array([x, y, z, 1.0], dtype=float)
        return p[:3]

    def _make_Tbt_start(self, T_board: np.ndarray, xb: float, yb: float, z: float) -> np.ndarray:
        R = tool_orientation_tilted()
        p_base = self._board_xyz_to_base(T_board, xb, yb, 0.0)
        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        T[:3, 3] = np.array([p_base[0], p_base[1], z], dtype=float)
        return T

    def plan_string(
        self,
        s: str,
        T_board: np.ndarray,
        *,
        theta_seed: np.ndarray | None = None,
    ) -> dict:
        """Plan a full string as a sequence of per-character plans (no execution)."""
        T_board = np.asarray(T_board, dtype=float)
        if T_board.shape != (4, 4):
            raise ValueError("plan_string: T_board must be 4x4.")

        seed = self.controller.theta_seed if theta_seed is None else as_6vec(theta_seed)

        x_cursor, y_cursor = 0.0, 0.0
        char_plans: list[dict] = []
        all_failures: list[dict] = []
        success = True

        for ch in s:
            if ch == "\r":
                continue
            if ch == "\n":
                x_cursor = 0.0
                y_cursor -= self.char_h + self.line_spacing
                continue
            if ch in (" ", "\t"):
                advance = 4.0 if ch == "\t" else 1.0
                x_cursor += (self.char_w + self.char_spacing) * advance
                continue

            T_start = self._make_Tbt_start(T_board, x_cursor, y_cursor, self.z0)
            plan = self.plan_character(T_start, ch, theta_seed=seed)
            char_plans.append(plan)

            if plan["theta_mod"]:
                seed = plan["theta_mod"][-1].copy()

            if not plan["success"]:
                success = False
                all_failures.extend(plan["failures"])
                if self.on_fail == "raise":
                    raise RuntimeError(f"plan_string: character {ch!r} failed.")

            x_cursor += self.char_w + self.char_spacing

        poses = [T for p in char_plans for T in p["poses"]]
        pen_down = [flag for p in char_plans for flag in p["pen_down"]]

        return {
            "string": s,
            "characters": char_plans,
            "poses": poses,
            "pen_down": pen_down,
            "failures": all_failures,
            "success": success,
        }

    def draw_string(
        self,
        s: str,
        T_board: np.ndarray,
        *,
        theta_seed: np.ndarray | None = None,
        move_to_first_pose: bool = True,
    ) -> dict:
        """Plan, optionally move-to-start, and execute the whole string."""
        T_board = np.asarray(T_board, dtype=float)
        if T_board.shape != (4, 4):
            raise ValueError("draw_string: T_board must be 4x4.")

        seed = self.controller.theta_seed if theta_seed is None else as_6vec(theta_seed)

        if move_to_first_pose:
            first_drawable = next((c for c in s if c not in (" ", "\t", "\n", "\r")), None)
            if first_drawable is not None:
                T_first = self._make_Tbt_start(T_board, 0.0, 0.0, self.z0)
                ok, theta_mod, _info = self.controller.move_to_pose(T_first, theta_seed=seed)
                if ok:
                    seed = theta_mod.copy()

        x_cursor, y_cursor = 0.0, 0.0
        char_outputs: list[dict] = []
        all_failures: list[dict] = []
        success = True

        for ch in s:
            if ch == "\r":
                continue
            if ch == "\n":
                x_cursor = 0.0
                y_cursor -= self.char_h + self.line_spacing
                continue
            if ch in (" ", "\t"):
                advance = 4.0 if ch == "\t" else 1.0
                x_cursor += (self.char_w + self.char_spacing) * advance
                continue

            T_start = self._make_Tbt_start(T_board, x_cursor, y_cursor, self.z0)
            out = self.draw_character(T_start, ch, theta_seed=seed)
            char_outputs.append(out)

            if out["theta_mod"]:
                seed = out["theta_mod"][-1].copy()

            if not out.get("success", True):
                success = False
                all_failures.extend(out.get("failures", []))
                if self.on_fail == "raise":
                    raise RuntimeError(f"draw_string: character {ch!r} failed.")

            x_cursor += self.char_w + self.char_spacing

        poses = [T for p in char_outputs for T in p["poses"]]
        pen_down = [flag for p in char_outputs for flag in p["pen_down"]]

        return {
            "string": s,
            "characters": char_outputs,
            "poses": poses,
            "pen_down": pen_down,
            "failures": all_failures,
            "success": success,
        }


def plot_plan_xy(
    plan: dict,
    *,
    title: str = "Planned pen-tip XY path (pen-down only)",
    color: str = "k",
    show: bool = True,
) -> None:
    """Plot the XY projection of pen-down segments from a plan/draw result."""
    poses: Sequence[np.ndarray] = plan.get("poses", [])
    if not poses:
        print("plot_plan_xy: no poses to plot.")
        return

    XY = np.array([T[:2, 3] for T in poses], dtype=float)
    pen = np.array(plan.get("pen_down", [True] * len(poses)), dtype=bool)

    plt.figure()
    i = 0
    while i < len(XY):
        if not pen[i]:
            i += 1
            continue
        j = i
        while j < len(XY) and pen[j]:
            j += 1
        plt.plot(XY[i:j, 0], XY[i:j, 1], color=color)
        i = j

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(title)
    if show:
        plt.show()


def _load_segments(filename: str | Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load (theta_start, theta_end) joint-angle pairs from a pickle file."""
    with open(filename, "rb") as f:
        data = pickle.load(f)

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for letter_idx, letter in enumerate(data):
        for seg_idx, seg in enumerate(letter):
            if not (isinstance(seg, (list, tuple)) and len(seg) == 2):
                raise ValueError(
                    f"Unrecognized segment at letter {letter_idx}, segment {seg_idx}: {type(seg)}"
                )
            start, end = seg
            segments.append((as_6vec(start), as_6vec(end)))
    return segments


def plot_pickle_word(
    filename: str | Path,
    kinematics: UR10e,
    *,
    ax: plt.Axes | None = None,
    show: bool = True,
) -> plt.Axes:
    """XY plot of the FK of every (start, end) joint pair in a Lab-1 pickle file."""
    segments = _load_segments(filename)
    if ax is None:
        _fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")

    for theta_start, theta_end in segments:
        p1 = kinematics.fk(theta_start)[0:2, 3]
        p2 = kinematics.fk(theta_end)[0:2, 3]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Projected word in XY plane")
    if show:
        plt.show()
    return ax


def execute_pickle_word(
    filename: str | Path,
    controller: RobotController,
    *,
    dedup_consecutive: bool = True,
) -> None:
    """Drive a Lab-1 pickle file of joint-angle pairs through ``controller.movej``."""
    segments = _load_segments(filename)
    last_q: np.ndarray | None = None

    for i, (q0, q1) in enumerate(segments):
        if (not dedup_consecutive) or (last_q is None) or (np.linalg.norm(q0 - last_q) > 1e-9):
            controller.movej(q0)
            last_q = q0.copy()
        if (not dedup_consecutive) or (np.linalg.norm(q1 - last_q) > 1e-9):
            controller.movej(q1)
            last_q = q1.copy()


class DrawRenderMode(str, Enum):
    """Rendering backend for ``run_draw``."""

    none = "none"
    viser = "viser"


def _build_draw_renderer(mode: DrawRenderMode, kin: UR10e) -> SimulationRenderer | None:
    if mode is DrawRenderMode.none:
        return None
    if mode is DrawRenderMode.viser:
        from .sim import ViserRenderer

        return ViserRenderer(kin)
    raise ValueError(f"Unknown render mode: {mode!r}")


def run_draw(
    text: Annotated[str, typer.Argument(help="Text to draw on the whiteboard.")] = "HELLO",
    ip: Annotated[str, typer.Option(help="IP address of the UR10e controller.")] = "192.168.0.2",
    simulate: Annotated[bool, typer.Option("--simulate/--no-simulate", help="Skip robot I/O.")] = True,
    dry_run: Annotated[bool, typer.Option("--dry-run/--no-dry-run", help="Connect to robot but print commands instead of executing.")] = False,
    render: Annotated[DrawRenderMode, typer.Option(help="Rendering backend: 'none' or 'viser'.")] = DrawRenderMode.none,
    step_duration: Annotated[float, typer.Option(help="Seconds per rendered joint step.")] = 0.2,
    board_x: Annotated[float, typer.Option(help="Board origin X (m) in base frame.")] = 0.40,
    board_y: Annotated[float, typer.Option(help="Board origin Y (m) in base frame.")] = -0.10,
    board_z: Annotated[float, typer.Option(help="Board origin Z (m) in base frame.")] = 0.0,
    z0: Annotated[float, typer.Option(help="Pen-up height (m) in base frame.")] = 0.10,
    z_touch: Annotated[float, typer.Option(help="Pen-down height (m) in base frame.")] = 0.05,
    tool_z_offset: Annotated[float, typer.Option(help="Distance from frame-6 to the pen tip (meters). 0 targets the flange; 0.30 matches Lab 2's default pen length.")] = 0.30,
) -> None:
    """Simulate / drive a UR10e to draw ``text`` on a whiteboard."""
    kin = UR10e(T6t=make_T([0.0, 0.0, float(tool_z_offset)]))
    renderer = _build_draw_renderer(render, kin)
    controller = RobotController(
        kin,
        simulate=simulate,
        dry_run=dry_run,
        renderer=renderer,
        step_duration=step_duration,
    )
    drawer = StringDrawer(controller, z0=z0, z_touch=z_touch)

    try:
        if not simulate:
            controller.connect(ip)

        T_home = np.eye(4, dtype=float)
        T_home[:3, :3] = tool_orientation_tilted()
        T_home[:3, 3] = [board_x, board_y, z0]
        controller.home(T_home, verify_fk=True)

        T_board = np.eye(4, dtype=float)
        T_board[:3, 3] = [board_x, board_y, board_z]

        if renderer is not None:
            # Size the board to roughly fit the string we're about to draw.
            line_width = (drawer.char_w + drawer.char_spacing) * max(1, len(text)) + 0.05
            line_height = drawer.char_h + 0.06
            drawer.add_board_visual(T_board, width=line_width, height=line_height)

        drawer.draw_string(text, T_board, move_to_first_pose=True)
    finally:
        controller.close()
        if renderer is not None:
            renderer.close(wait=(render is DrawRenderMode.viser))
