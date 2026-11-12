"""End-effector tracks. Every Cartesian track is checked for reachability first (exhaustive multi-start DLS),
so what the reviewer sees is solver quality, not the robot's workspace limits."""

import numpy as np
import torch

from .kinematics import KinChain, chains_to_batch, fk, pose_errors
from .solver import dls_polish

TRACKS = ["figure8", "circle", "square", "line", "helix", "random_poses", "manual"]
TRACK_HELP = {
    "figure8": "Cartesian figure-8, fixed tool orientation",
    "circle": "Cartesian circle, fixed tool orientation",
    "square": "Cartesian square (sharp corners), fixed orientation",
    "line": "straight line back and forth",
    "helix": "3-turn helix up and down",
    "random_poses": "jump to a new random reachable pose every 2 s (tests solving from far away)",
    "manual": "drive the target with the 'target dx/dy/dz/roll/pitch/yaw' sliders",
}


def comfortable_pose(chain: KinChain, rng):
    mid, half = 0.5 * (chain.lower + chain.upper), 0.5 * (chain.upper - chain.lower)
    q0 = np.clip(mid + rng.uniform(-0.3, 0.3, chain.dof) * np.minimum(half, 1.5), chain.lower, chain.upper)
    return q0, chain.fk_np(q0)


def shape_offsets(kind, n):
    """Unit-size offsets (n,3) in the base frame; every shape starts at 0 and is closed."""
    t = 2 * np.pi * np.arange(n) / n
    z = np.zeros(n)
    if kind == "figure8":
        return np.stack([z, np.sin(t), 0.5 * np.sin(2 * t)], 1)
    if kind == "circle":
        return np.stack([z, 0.5 * np.sin(t), 0.5 * (1 - np.cos(t))], 1)
    if kind == "line":
        return np.stack([z, np.sin(t), z], 1)
    if kind == "square":
        s = (np.arange(n) / n) * 4
        corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], float) - [0.5, 0]
        k = np.minimum(s.astype(int), 3); f = (s - k)[:, None]
        yz = corners[k] * (1 - f) + corners[k + 1] * f
        yz = yz - corners[0]
        return np.stack([z, yz[:, 0] * 0.8, yz[:, 1] * 0.8], 1)
    if kind == "helix":
        s = np.arange(n) / n
        ang = 6 * np.pi * s
        h = 0.8 * (1 - np.abs(2 * s - 1))                   # up then down
        return np.stack([0.35 * (1 - np.cos(ang)), 0.35 * np.sin(ang), h], 1)
    raise ValueError(kind)


@torch.no_grad()
def reachable(chain, goals, starts=24, iters=40, tol_mm=1.0, max_check=40):
    sel = goals[:: max(1, len(goals) // max_check)]
    n = len(sel)
    cb = chains_to_batch([chain] * (n * starts))
    G = torch.tensor(np.stack(sel), dtype=torch.float32).repeat_interleave(starts, 0)
    q = dls_polish(cb, cb.sample_q(), G, iters)
    T, _, _ = fk(cb, q); pe, _ = pose_errors(T, G)
    return bool((pe.view(n, starts).min(1).values * 1e3 < tol_mm).all())


def build_track(chain: KinChain, kind, n=240, size=1.0, seed=0, attempts=8):
    """Returns (q_start, goals[list of 4x4], note). `size` scales the default 12%-of-reach track."""
    rng = np.random.default_rng(seed)
    if kind == "random_poses":
        hold = max(1, n // 8)
        goals = []
        for _ in range(8):
            q = rng.uniform(chain.lower, chain.upper)
            goals += [chain.fk_np(q)] * hold
        q0, _ = comfortable_pose(chain, rng)
        return q0, goals, "random reachable poses"
    if kind == "manual":
        q0, T0 = comfortable_pose(chain, rng)
        return q0, [T0], "manual target"
    off = shape_offsets(kind, n)
    for att in range(attempts):
        q0, T0 = comfortable_pose(chain, rng)
        a = 0.12 * size * chain.reach * (0.75 ** (att // 2))
        goals = []
        for d in off:
            T = T0.copy(); T[:3, 3] = T0[:3, 3] + a * d; goals.append(T)
        if reachable(chain, goals):
            note = f"{kind}, size {a * 100:.1f} cm" + ("" if att < 2 else " (shrunk to stay reachable)")
            return q0, goals, note
    # fallback: a smooth joint-space path is reachable by construction
    mid, half = 0.5 * (chain.lower + chain.upper), 0.5 * (chain.upper - chain.lower)
    ph, fr = rng.uniform(0, 2 * np.pi, chain.dof), rng.integers(1, 3, chain.dof)
    q0 = np.clip(mid + rng.uniform(-0.3, 0.3, chain.dof) * half, chain.lower, chain.upper)
    qs = [np.clip(q0 + 0.4 * size * np.sin(2 * np.pi * fr * k / n + ph), chain.lower, chain.upper) for k in range(n)]
    return qs[0], [chain.fk_np(q) for q in qs], f"{kind} not reachable here -> smooth joint-space path"


def manual_goal(T0, dx, dy, dz, droll, dpitch, dyaw, reach):
    from .kinematics import rpy_to_R
    T = T0.copy()
    T[:3, 3] = T0[:3, 3] + np.array([dx, dy, dz]) * reach
    T[:3, :3] = rpy_to_R([droll, dpitch, dyaw]) @ T0[:3, :3]
    return T
