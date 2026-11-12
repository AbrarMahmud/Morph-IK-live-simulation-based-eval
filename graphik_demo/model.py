"""The URDF-conditioned graph IK network (proposal + refinement trunks).

Generated from the training notebook (ik-v8-urdf-graph-ik.ipynb) so the demo runs exactly the
trained architecture. Do not edit by hand; regenerate instead."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv

from .kinematics import NMAX, ChainBatch, fk, rot6d, rot_log

USE_AMP = False                   # inference in fp32
AMP_DTYPE = torch.float16


def static_node_features(cb: ChainBatch):
    B, N = cb.mask.shape
    s = cb.reach[:, None, None]
    t = cb.origin[:, :, :3, 3] / s
    n = cb.mask.sum(1, keepdim=True)
    i = torch.arange(N, device=cb.mask.device)[None].float().expand(B, N)
    pos = i / (n - 1).clamp_min(1)
    return torch.cat([
        t, rot6d(cb.origin[:, :, :3, :3]), cb.axis,
        (cb.lower / math.pi)[..., None], (cb.upper / math.pi)[..., None], (cb.half / math.pi)[..., None],
        t.norm(dim=-1, keepdim=True), pos[..., None], (1 - pos)[..., None],
        (i == 0).float()[..., None], (i == n - 1).float()[..., None],
        (n / NMAX).expand(B, N)[..., None]], -1) * cb.mask[..., None]


def static_edge_features(cb: ChainBatch):
    """Edge i->i+1 carries the rigid transform from joint i to joint i+1 (the 'link')."""
    s = cb.reach[:, None, None]
    child = cb.origin[:, 1:]
    return torch.cat([child[..., :3, 3] / s, rot6d(child[..., :3, :3])], -1)


def global_static_features(cb: ChainBatch):
    return torch.cat([cb.tool[:, :3, 3] / cb.reach[:, None], rot6d(cb.tool[:, :3, :3]),
                      cb.mask.sum(1, keepdim=True) / NMAX], -1)


def dynamic_features(cb: ChainBatch, q, goal):
    """Everything a Jacobian step would use, expressed in base frame and normalised by reach,
    so the network learns a *geometric* update rule that transfers across robots."""
    Tee, P, Z = fk(cb, q)
    s = cb.reach[:, None, None]
    pe, pg = Tee[:, :3, 3], goal[:, :3, 3]
    e_pos = (pg - pe) / cb.reach[:, None]                                   # (B,3)
    e_rot = rot_log(goal[:, :3, :3] @ Tee[:, :3, :3].transpose(-1, -2))     # (B,3) world-frame rot error
    jp = torch.cross(Z, (pe[:, None] - P), dim=-1) / s                      # position Jacobian column
    u = cb.q_to_u(q)
    node = torch.cat([
        P / s, Z, (pg[:, None] - P) / s, (pe[:, None] - P) / s, jp,
        (jp * e_pos[:, None]).sum(-1, keepdim=True) * 10,                    # J^T e  (position part)
        (Z * e_rot[:, None]).sum(-1, keepdim=True),                         # J^T e  (rotation part)
        torch.sin(q)[..., None], torch.cos(q)[..., None], u[..., None],
        ((q - cb.lower) / (cb.upper - cb.lower).clamp_min(1e-3))[..., None]], -1) * cb.mask[..., None]   # 21
    glob = torch.cat([e_pos * 10, e_pos.norm(dim=-1, keepdim=True) * 10, e_rot,
                      e_rot.norm(dim=-1, keepdim=True), rot6d(goal[:, :3, :3]), pg / cb.reach[:, None],
                      rot6d(Tee[:, :3, :3])], -1)                                                       # 23
    return node, glob


NODE_IN = 21 + 21


EDGE_IN = 9 + 2


GLOB_IN = 10 + 23 + 1


def build_graph(cb: ChainBatch, node_dense):
    """Dense (B,N,F) -> PyG flat graph (chain edges both directions)."""
    m = cb.mask.bool()
    B, N = m.shape
    flat_id = (torch.cumsum(m.view(-1).long(), 0) - 1).view(B, N)
    x = node_dense[m]
    batch = torch.arange(B, device=m.device)[:, None].expand(B, N)[m]
    e_ok = m[:, :-1] & m[:, 1:]
    src, dst = flat_id[:, :-1][e_ok], flat_id[:, 1:][e_ok]
    ef = static_edge_features(cb)[e_ok]
    one = torch.ones_like(src, dtype=ef.dtype)[:, None]
    edge_index = torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], 1)
    edge_attr = torch.cat([torch.cat([ef, one, 0 * one], -1), torch.cat([ef, 0 * one, one], -1)], 0)
    return x, edge_index, edge_attr, batch, m


def mlp(i, h, o, drop=0.0):
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Dropout(drop), nn.Linear(h, o))


class GraphIKBlock(nn.Module):
    """GPS-style block: GINE (chain-local, edge = link transform) + MHA over [goal token, all joints]."""

    def __init__(self, H, heads=4, drop=0.0, use_local=True, use_attn=True):
        super().__init__()
        self.use_local, self.use_attn = use_local, use_attn
        self.ln1, self.ln2, self.ln3, self.lng = nn.LayerNorm(H), nn.LayerNorm(H), nn.LayerNorm(H), nn.LayerNorm(H)
        self.local = GINEConv(mlp(H, H, H), edge_dim=H, train_eps=True)
        self.attn = nn.MultiheadAttention(H, heads, dropout=drop, batch_first=True)
        self.ff = mlp(H, 2 * H, H, drop)
        self.ffg = mlp(H, 2 * H, H, drop)
        self.film = nn.Linear(H, 2 * H)
        self.drop = nn.Dropout(drop)

    def forward(self, h, g, edge_index, e, m):
        B, N = m.shape
        if self.use_local:     # chain-local message passing (edge = link transform)
            h = h + self.drop(self.local(self.ln1(h), edge_index, e))
        if self.use_attn:      # whole-arm attention with the goal token
            hn = self.ln2(h); dense = hn.new_zeros(B, N, hn.shape[-1]); dense[m] = hn
            seq = torch.cat([self.lng(g)[:, None], dense], 1)
            kpm = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=m.device), ~m], 1)
            a, _ = self.attn(seq, seq, seq, key_padding_mask=kpm, need_weights=False)
            g = g + self.drop(a[:, 0]); h = h + self.drop(a[:, 1:][m])
        else:                  # keep a (mean-pooled) robot summary flowing into the goal token
            dense = h.new_zeros(B, N, h.shape[-1]); dense[m] = h
            g = g + (dense.sum(1) / m.sum(1, keepdim=True).clamp_min(1)).to(g.dtype)
        sc, sh = self.film(g).chunk(2, -1)                         # goal/robot-conditioned FiLM
        batch = torch.arange(B, device=m.device)[:, None].expand(B, N)[m]
        h = h + self.ff(self.ln3(h) * (1 + sc[batch]) + sh[batch])
        g = g + self.ffg(g)
        return h, g


class GraphIKTrunk(nn.Module):
    def __init__(self, H=192, layers=6, heads=4, text_dim=0, drop=0.0, text_dropout=0.3, arch="gnn"):
        super().__init__()
        self.arch = arch
        self.node_enc = mlp(NODE_IN, H, H)
        self.edge_enc = mlp(EDGE_IN, H, H)
        self.glob_enc = mlp(GLOB_IN, H, H)
        self.text_proj = nn.Sequential(nn.Linear(text_dim, H), nn.SiLU(), nn.Linear(H, H)) if text_dim > 0 else None
        self.text_dropout = text_dropout
        use_local = arch in ("gnn", "no_attn")
        use_attn = arch in ("gnn", "no_local")
        self.blocks = nn.ModuleList([GraphIKBlock(H, heads, drop, use_local, use_attn) for _ in range(layers)])
        self.out_ln = nn.LayerNorm(H)

    def forward(self, cb, q, goal, has_seed):
        nd, gd = dynamic_features(cb, q, goal)
        node = torch.cat([static_node_features(cb), nd], -1)
        x, ei, ea, batch, m = build_graph(cb, node)
        g = torch.cat([global_static_features(cb), gd, has_seed[:, None].float()], -1)
        # FK + features above stay fp32 (geometry needs precision); the network below may run in fp16
        with torch.autocast(device_type=x.device.type, dtype=AMP_DTYPE, enabled=USE_AMP):
            h, e, g = self.node_enc(x), self.edge_enc(ea), self.glob_enc(g)
            if self.text_proj is not None and cb.text is not None:
                t = self.text_proj(cb.text)
                if self.training:   # modality dropout: geometry must suffice on its own
                    t = t * (torch.rand(t.shape[0], 1, device=t.device) > self.text_dropout).float()
                g = g + t
            for blk in self.blocks:
                h, g = blk(h, g, ei, e, m)
            h = self.out_ln(h)
        return h.float(), g.float(), m, batch


class MLPTrunk(nn.Module):
    """Ablation: same features, NO graph. Joints are concatenated into one padded vector (fixed NMAX slots),
    so the network cannot share weights across joints or exploit the kinematic structure."""

    def __init__(self, H=192, layers=6, heads=4, text_dim=0, drop=0.0, text_dropout=0.3, arch="mlp"):
        super().__init__()
        self.text_proj = nn.Sequential(nn.Linear(text_dim, H), nn.SiLU(), nn.Linear(H, H)) if text_dim > 0 else None
        self.text_dropout = text_dropout
        W = 4 * H
        self.inp = nn.Linear(NMAX * NODE_IN + NMAX * EDGE_IN + GLOB_IN, W)
        self.body = nn.ModuleList([nn.Sequential(nn.LayerNorm(W), mlp(W, W, W, drop)) for _ in range(max(1, layers // 2))])
        self.out_nodes = nn.Linear(W, NMAX * H); self.out_g = nn.Linear(W, H)
        self.out_ln = nn.LayerNorm(H); self.H = H

    def forward(self, cb, q, goal, has_seed):
        nd, gd = dynamic_features(cb, q, goal)
        node = torch.cat([static_node_features(cb), nd], -1)                  # (B,N,F)
        B, N, _ = node.shape
        ef = torch.cat([static_edge_features(cb), torch.zeros(B, 1, 9, device=node.device)], 1) * cb.mask[..., None]
        ef = torch.cat([ef, torch.zeros(B, N, 2, device=node.device)], -1)
        g = torch.cat([global_static_features(cb), gd, has_seed[:, None].float()], -1)
        z = torch.cat([node.reshape(B, -1), ef.reshape(B, -1), g], -1)
        m = cb.mask.bool()
        batch = torch.arange(B, device=m.device)[:, None].expand(B, N)[m]
        with torch.autocast(device_type=z.device.type, dtype=AMP_DTYPE, enabled=USE_AMP):
            z = self.inp(z)
            if self.text_proj is not None and cb.text is not None:
                t = self.text_proj(cb.text)
                if self.training:
                    t = t * (torch.rand(t.shape[0], 1, device=t.device) > self.text_dropout).float()
                z = z + t.repeat(1, 4)
            for blk in self.body:
                z = z + blk(z)
            h = self.out_ln(self.out_nodes(z).view(B, N, self.H))[m]
            gg = self.out_g(z)
        return h.float(), gg.float(), m, batch


def build_model(cfg, text_dim):
    return URDFGraphIK(cfg["H"], cfg["layers"], cfg["heads"], cfg["K"], text_dim, arch=cfg.get("arch", "gnn"))


class URDFGraphIK(nn.Module):
    """Two cooperating GNNs over the robot's kinematic graph.

    * proposal net : goal -> K whole-arm hypotheses (MDN whose mixture component is SHARED by all joints,
                     so each component is one coherent IK branch: elbow up/down, wrist flip ...)
    * refine net   : (goal, current q) -> delta q, applied iteratively with exact FK in the loop.
    The previous joint state is just one possible starting point for the refiner; the answer always comes
    from closing the loop through FK, never from copying the input."""

    def __init__(self, H=192, layers=6, heads=4, K=8, text_dim=0, drop=0.0, arch="gnn"):
        super().__init__()
        self.K = K
        Trunk = MLPTrunk if arch == "mlp" else GraphIKTrunk
        self.prop = Trunk(H, layers, heads, text_dim, drop, arch=arch)
        self.ref = Trunk(H, layers, heads, text_dim, drop, arch=arch)
        self.mu = mlp(H, H, K); self.logsig = mlp(H, H, K); self.logit = mlp(H, H, K)
        self.dq = mlp(H, H, 1)
        nn.init.zeros_(self.dq[-1].weight); nn.init.zeros_(self.dq[-1].bias)
        nn.init.constant_(self.logsig[-1].bias, -1.5)

    def q_ref(self, cb):
        return cb.clamp_q(torch.zeros_like(cb.mask))

    def propose(self, cb, goal):
        q0 = self.q_ref(cb)
        h, g, m, _ = self.prop(cb, q0, goal, torch.zeros(cb.B, device=goal.device))
        B, N = m.shape
        mu = h.new_zeros(B, N, self.K); ls = h.new_zeros(B, N, self.K)
        mu[m] = self.mu(h); ls[m] = self.logsig(h)
        sig = F.softplus(ls) + 2e-3
        return self.logit(g), mu, sig            # (B,K), (B,N,K) in normalised u-space

    def refine_step(self, cb, q, goal, has_seed=None):
        if has_seed is None: has_seed = torch.ones(cb.B, device=q.device)
        h, g, m, _ = self.ref(cb, q, goal, has_seed)
        du = q.new_zeros(q.shape); du[m] = torch.tanh(self.dq(h)[:, 0])
        return cb.clamp_q(q + du * cb.half)

    @staticmethod
    def mdn_nll(logit, mu, sig, u, mask):
        """-log sum_k pi_k prod_i N(u_i | mu_ik, sig_ik)   (joint-coherent mixture)."""
        lp = (-0.5 * ((u[..., None] - mu) / sig) ** 2 - sig.log() - 0.5 * math.log(2 * math.pi))
        lp = (lp * mask[..., None]).sum(1)                            # (B,K)
        return -(torch.logsumexp(F.log_softmax(logit, -1) + lp, -1) / mask.sum(1)).mean()
