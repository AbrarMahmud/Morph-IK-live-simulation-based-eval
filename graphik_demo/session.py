"""One live IK session: a robot, its current joint state, and a solve() per frame."""
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .kinematics import NMAX, KinChain, chains_to_batch, describe_chain, fk, pose_errors
from .solver import solve_ik


class IKSession:
    def __init__(self, model, text_encoder=None):
        self.model, self.text_encoder = model, text_encoder
        self.chain, self.cb, self.q = None, None, None

    def set_robot(self, chain: KinChain, q_start=None):
        self.chain = chain
        self.cb = chains_to_batch([chain])
        if self.text_encoder is not None and self.model.ref.text_proj is not None:
            self.cb.text = torch.tensor(self.text_encoder([describe_chain(chain)]), dtype=torch.float32)
        self.reset(q_start)

    def reset(self, q_start=None):
        q = np.zeros(self.chain.dof) if q_start is None else np.asarray(q_start, dtype=float)
        self.q = np.clip(q, self.chain.lower, self.chain.upper)

    def _pad(self, v):
        return F.pad(torch.as_tensor(np.asarray(v), dtype=torch.float32), (0, NMAX - self.chain.dof))[None]

    @torch.no_grad()
    def solve(self, goal_T, use_seed=True, polish=True, refine_iters=None, polish_iters=None):
        """Solve one frame. use_seed=False ignores the current joints (pure learned IK from scratch).

        Iteration budget (unless given): small "tracking" budget (2 refinement / 10 polish steps) while the target
        moves smoothly; the full budget used in the paper evaluation (4 / 20) when the target jumps far or no
        seed is used."""
        goal = torch.tensor(goal_T, dtype=torch.float32)[None]
        seed = self._pad(self.q) if use_seed else None
        T_now = self.chain.fk_np(self.q)
        far = (np.linalg.norm(T_now[:3, 3] - goal_T[:3, 3]) > 0.05 * self.chain.reach
               or np.trace(T_now[:3, :3].T @ goal_T[:3, :3]) < 1 + 2 * math.cos(math.radians(15)))
        full = far or not use_seed
        refine_iters = refine_iters if refine_iters is not None else (4 if full else 2)
        polish_iters = polish_iters if polish_iters is not None else (20 if full else 10)
        t0 = time.perf_counter()
        q, q_net, _ = solve_ik(self.model, self.cb, goal, seed, refine_iters=refine_iters,
                               polish_iters=polish_iters if polish else 0)
        if not polish:
            q = q_net
        ms = 1e3 * (time.perf_counter() - t0)
        T, _, _ = fk(self.cb, q)
        pe, re = pose_errors(T, goal)
        qn = q[0, :self.chain.dof].numpy().astype(float)
        jump = math.degrees(np.abs(qn - self.q).max())
        self.q = qn
        return qn, dict(pos_err_mm=pe.item() * 1e3, rot_err_deg=math.degrees(re.item()), ms=ms, jump_deg=jump,
                        ee=T[0].numpy(), budget="full" if full else "tracking")
