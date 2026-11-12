"""Batched IK solve: learned proposals -> learned refinement -> damped-least-squares polish.

Generated from the training notebook (ik-v8-urdf-graph-ik.ipynb) so the demo runs exactly the
trained architecture. Do not edit by hand; regenerate instead."""
import torch

from .kinematics import fk, pose_loss, rot_log


def dls_polish(cb, q, goal, iters=20, lam=0.02, max_step=0.4):
    """Damped least-squares with per-sample adaptive damping (Levenberg-Marquardt style). Uses the same
    differentiable FK; only accepts improving steps, so it can never make a network solution worse."""
    B, N = q.shape
    lam = torch.full((B,), lam, device=q.device)
    s = cb.reach[:, None]
    def err(q):
        Tee, P, Z = fk(cb, q)
        e = torch.cat([(goal[:, :3, 3] - Tee[:, :3, 3]) / s, rot_log(goal[:, :3, :3] @ Tee[:, :3, :3].transpose(-1, -2))], -1)
        return e, Tee, P, Z
    e, Tee, P, Z = err(q)
    for _ in range(iters):
        Jp = torch.cross(Z, Tee[:, None, :3, 3] - P, dim=-1) / s[..., None]
        J = (torch.cat([Jp, Z], -1) * cb.mask[..., None]).transpose(1, 2)         # (B,6,N)
        A = J @ J.transpose(1, 2) + (lam ** 2)[:, None, None] * torch.eye(6, device=q.device)
        dq = (J.transpose(1, 2) @ torch.linalg.solve(A, e[..., None]))[..., 0]
        dq = dq.clamp(-max_step, max_step)
        q_new = cb.clamp_q(q + dq)
        e_new, T_new, P_new, Z_new = err(q_new)
        better = e_new.norm(dim=-1) < e.norm(dim=-1)
        bm = better[:, None]
        q = torch.where(bm, q_new, q); e = torch.where(bm, e_new, e)
        Tee = torch.where(bm[..., None], T_new, Tee); P = torch.where(bm[..., None], P_new, P); Z = torch.where(bm[..., None], Z_new, Z)
        lam = torch.where(better, lam * 0.5, lam * 4).clamp(1e-4, 10)
    return q


@torch.no_grad()
def solve_ik(model, cb, goal, q_seed=None, refine_iters=4, polish_iters=20, n_samples=0, accept_tol=2e-4, seed_weight=2e-3,
             use_proposer=True, seed_fallback=True, return_all=False):
    """Batched IK. Returns best q (B,N) after polish, best network-only q, and the chosen candidate index.
    Candidates: [seed (if given)] + K proposal means (+ optional MDN samples), all refined in parallel."""
    model.eval()
    B, N = cb.mask.shape
    if use_proposer:
        logit, mu, sig = model.propose(cb, goal)
        us = [mu.permute(0, 2, 1)]                                                    # (B,K,N)
        if n_samples > 0:
            comp = torch.distributions.Categorical(logits=logit).sample((n_samples,)).T  # (B,S)
            m_ = torch.gather(mu.permute(0, 2, 1), 1, comp[..., None].expand(-1, -1, N))
            s_ = torch.gather(sig.permute(0, 2, 1), 1, comp[..., None].expand(-1, -1, N))
            us.append(m_ + s_ * torch.randn_like(m_))
        u = torch.cat(us, 1)
        q_c = (cb.mid[:, None] + cb.half[:, None] * u.clamp(-1, 1)) * cb.mask[:, None]
    else:                      # ablation "refiner only": start from the home pose
        q_c = model.q_ref(cb)[:, None]
    flags = torch.zeros(B, q_c.shape[1], device=goal.device)
    frozen = torch.zeros(B, q_c.shape[1], dtype=torch.bool, device=goal.device)
    if q_seed is not None and seed_fallback:
        # seed enters twice: refined by the GNN, and untouched (DLS-only) = classic tracking fallback
        q_c = torch.cat([q_seed[:, None], q_seed[:, None], q_c], 1)
        flags = torch.cat([torch.ones(B, 2, device=goal.device), flags], 1)
        frozen = torch.cat([torch.tensor([[False, True]], device=goal.device).expand(B, 2), frozen], 1)
    elif q_seed is not None:   # network-only mode: seed is only refined by the GNN
        q_c = torch.cat([q_seed[:, None], q_c], 1)
        flags = torch.cat([torch.ones(B, 1, device=goal.device), flags], 1)
        frozen = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=goal.device), frozen], 1)
    C = q_c.shape[1]
    cbC, goalC = cb.repeat(C), goal.repeat_interleave(C, 0)
    q = q_c.reshape(B * C, N)
    fz = frozen.reshape(-1, 1)
    for _ in range(refine_iters):
        q = torch.where(fz, q, model.refine_step(cbC, q, goalC, flags.reshape(-1)))
    q_net = q.clone()
    if polish_iters > 0:
        q = dls_polish(cbC, q, goalC, polish_iters)

    def pick(qq):
        T, _, _ = fk(cbC, qq)
        sc = pose_loss(T, goalC, cbC.reach).view(B, C)
        if q_seed is not None:
            # lexicographic: among candidates that reach the pose, take the one closest to the current
            # state (no IK-branch flips during tracking); otherwise the most accurate one.
            # Candidates within accept_tol count as equally accurate -> the one closest to the current state wins.
            # Beyond it, accuracy is traded against motion: a 1 rad branch flip costs `seed_weight` of normalised
            # pose error (2e-3 ~ 2 mm on a 1 m arm). E.g. at a joint limit the arm accepts ~1 mm for one frame
            # instead of flipping 140 degrees to another IK branch.
            d = (qq.view(B, C, N) - q_seed[:, None]).abs().amax(-1)
            sc = (sc - accept_tol).clamp_min(0) + seed_weight * d
        k = sc.argmin(1)
        return qq.view(B, C, N)[torch.arange(B), k], k
    q_best, k = pick(q)
    q_net_best, _ = pick(q_net)
    if return_all:
        return q_best, q_net_best, k, q.view(B, C, N), cbC, goalC
    return q_best, q_net_best, k
