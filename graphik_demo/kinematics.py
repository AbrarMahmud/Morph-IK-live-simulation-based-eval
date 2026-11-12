"""URDF parsing, kinematic chains and differentiable batched forward kinematics.

Generated from the training notebook (ik-v8-urdf-graph-ik.ipynb) so the demo runs exactly the
trained architecture. Do not edit by hand; regenerate instead."""
import os, math, copy, warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np
import torch

DEVICE = torch.device("cpu")      # the demo runs on CPU (set by graphik_demo.hub.load_model if a GPU is wanted)
NMAX = 8
ACTUATED = ("revolute", "continuous")
EE_NAME_PRIORITY = ("tool0", "ee_link", "end_effector", "flange", "panda_hand", "tool", "tcp")


def rpy_to_R(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    # URDF convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def R_to_rpy(R):
    pitch = math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0]))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


def make_T(R=None, t=None):
    T = np.eye(4)
    if R is not None: T[:3, :3] = R
    if t is not None: T[:3, 3] = t
    return T


@dataclass
class KinChain:
    """Serial chain between base link and EE link.

    Fixed / excluded / non-actuated joints are folded into the next actuated joint's
    origin, trailing fixed joints become the tool transform.  Joint i frame = parent
    frame @ origins[i]; the joint then rotates about axes[i] (joint frame)."""
    name: str
    origins: np.ndarray            # (N,4,4)
    axes: np.ndarray               # (N,3) unit
    lower: np.ndarray              # (N,)
    upper: np.ndarray              # (N,)
    tool: np.ndarray               # (4,4)
    joint_names: List[str]
    base_link: str = ""
    ee_link: str = ""
    source: str = ""               # urdf path (for pybullet checks)
    family: str = ""               # robot this one was derived from

    @property
    def dof(self): return len(self.joint_names)

    @property
    def reach(self):
        return float(np.linalg.norm(self.origins[:, :3, 3], axis=-1).sum() + np.linalg.norm(self.tool[:3, 3]) + 1e-6)

    def fk_np(self, q):
        T = np.eye(4)
        for i in range(self.dof):
            T = T @ self.origins[i]
            a = self.axes[i]; K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
            Rq = np.eye(3) + math.sin(q[i]) * K + (1 - math.cos(q[i])) * K @ K
            T = T @ make_T(Rq)
        return T @ self.tool


def _origin(el):
    o = el.find("origin")
    xyz = [0.0, 0.0, 0.0]; rpy = [0.0, 0.0, 0.0]
    if o is not None:
        if o.get("xyz"): xyz = [float(v) for v in o.get("xyz").split()]
        if o.get("rpy"): rpy = [float(v) for v in o.get("rpy").split()]
    return make_T(rpy_to_R(rpy), np.array(xyz))


def _load_urdf_xml(path):
    if path.endswith(".xacro"):
        try:
            import xacro  # pip install xacro
            return ET.fromstring(xacro.process_file(path).toxml())
        except ImportError:
            warnings.warn(f"{path}: xacro package missing, parsing raw XML (macros will NOT be expanded)")
    return ET.parse(path).getroot()


def resolve_pybullet_indices(path, indices):
    """The original notebook excluded joints by PyBullet index -> convert to joint names."""
    import pybullet as p, pybullet_data
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
    rid = p.loadURDF(path, useFixedBase=True, physicsClientId=cid)
    names = [p.getJointInfo(rid, i, physicsClientId=cid)[1].decode() for i in indices]
    p.disconnect(cid)
    return names


def parse_urdf_chain(path: str, ee_link: Optional[str] = None, base_link: Optional[str] = None,
                     exclude: Sequence[Union[int, str]] = (), name: Optional[str] = None) -> KinChain:
    """Read exact joint origins / axes / limits from the URDF itself.

    `exclude` may hold joint names or PyBullet joint indices; excluded joints are frozen at 0."""
    root = _load_urdf_xml(path)
    links = [l.get("name") for l in root.findall("link")]
    joints = {}
    for j in root.findall("joint"):
        ax = j.find("axis")
        axis = np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else np.array([1.0, 0, 0])
        lim = j.find("limit")
        lo = float(lim.get("lower", 0)) if lim is not None else 0.0
        hi = float(lim.get("upper", 0)) if lim is not None else 0.0
        jtype = j.get("type")
        if j.find("mimic") is not None:
            jtype = "fixed"            # mimic joints are not independent DOF
        joints[j.get("name")] = dict(name=j.get("name"), type=jtype, parent=j.find("parent").get("link"),
                                     child=j.find("child").get("link"), T=_origin(j),
                                     axis=axis / (np.linalg.norm(axis) + 1e-12), lo=lo, hi=hi)
    excl = set(e for e in exclude if isinstance(e, str))
    ints = [e for e in exclude if isinstance(e, (int, np.integer))]
    if ints:
        excl |= set(resolve_pybullet_indices(path, ints))

    child_to_joint = {j["child"]: j for j in joints.values()}
    kids = {}
    for j in joints.values(): kids.setdefault(j["parent"], []).append(j)
    roots = [l for l in links if l not in child_to_joint]
    base = base_link or roots[0]

    def path_to(link):
        out = []
        while link in child_to_joint and link != base:
            j = child_to_joint[link]; out.append(j); link = j["parent"]
        return out[::-1]

    def n_act(pth): return sum(j["type"] in ACTUATED and j["name"] not in excl for j in pth)

    if ee_link is None:
        for cand in EE_NAME_PRIORITY:
            if cand in links: ee_link = cand; break
    if ee_link is None:
        # deepest actuated chain, then extend through fixed joints only (avoids gripper fingers)
        leaves = [l for l in links if l not in kids]
        best = max(leaves, key=lambda l: (n_act(path_to(l)), -len(path_to(l))))
        pth = path_to(best)
        last = max(i for i, j in enumerate(pth) if j["type"] in ACTUATED and j["name"] not in excl)
        link = pth[last]["child"]
        while True:
            fixed_kids = [j for j in kids.get(link, []) if j["type"] == "fixed"]
            if not fixed_kids: break
            link = fixed_kids[0]["child"]
        ee_link = link

    origins, axes, lo, hi, names = [], [], [], [], []
    T_acc = np.eye(4)
    for j in path_to(ee_link):
        if j["type"] in ACTUATED and j["name"] not in excl:
            origins.append(T_acc @ j["T"]); axes.append(j["axis"]); names.append(j["name"])
            l, h = j["lo"], j["hi"]
            if j["type"] == "continuous" or np.isclose(l, h):
                l, h = -math.pi, math.pi
            lo.append(l); hi.append(h)
            T_acc = np.eye(4)
        else:  # fixed / prismatic / excluded -> frozen at zero
            T_acc = T_acc @ j["T"]
    if not names:
        raise ValueError(f"{path}: no actuated revolute joints between {base} and {ee_link}")
    nm = name or os.path.splitext(os.path.basename(path))[0]
    return KinChain(nm, np.stack(origins), np.stack(axes), np.array(lo), np.array(hi), T_acc,
                    names, base, ee_link, source=path, family=nm)


def perturb_chain(ch: KinChain, rng: np.random.Generator, scale=(0.7, 1.35), jitter=0.02,
                  ang_jitter=0.0, tag="var") -> KinChain:
    """Same topology / axes, different link lengths (per-link scaling) + small offsets."""
    c = copy.deepcopy(ch)
    for i in range(c.dof):
        t = c.origins[i, :3, 3]
        c.origins[i, :3, 3] = t * rng.uniform(*scale) + rng.normal(0, jitter, 3) * (np.abs(t) > 1e-4)
        if ang_jitter > 0:
            c.origins[i, :3, :3] = c.origins[i, :3, :3] @ rpy_to_R(rng.normal(0, ang_jitter, 3))
    tt = c.tool[:3, 3]
    c.tool[:3, 3] = tt * rng.uniform(*scale)
    c.name = f"{ch.name}_{tag}"; c.source = ""
    return c


def describe_chain(ch: KinChain) -> str:
    parts = []
    for i in range(ch.dof):
        t = ch.origins[i, :3, 3]; rpy = R_to_rpy(ch.origins[i, :3, :3])
        parts.append(f"Joint {i} '{ch.joint_names[i]}' is revolute about axis {np.round(ch.axes[i], 3).tolist()}, "
                     f"range {ch.lower[i]:.3f} to {ch.upper[i]:.3f} rad, offset {np.round(t, 4).tolist()} m, "
                     f"twist {np.round(rpy, 3).tolist()} rad, link length {np.linalg.norm(t):.3f} m")
    parts.append(f"Tool offset {np.round(ch.tool[:3, 3], 4).tolist()} m")
    return f"{ch.dof}-DOF serial arm, reach {ch.reach:.3f} m. " + " | ".join(parts)


class ChainBatch:
    """Dense padded robot description; joints are left-aligned, padding = identity / q=0."""
    FIELDS = ("origin", "axis", "lower", "upper", "mask", "tool", "reach", "text", "nidx")

    def __init__(self, **kw):
        for k in self.FIELDS: setattr(self, k, kw.get(k))

    def __getitem__(self, idx):
        return ChainBatch(**{k: (getattr(self, k)[idx] if getattr(self, k) is not None else None) for k in self.FIELDS})

    def to(self, dev):
        return ChainBatch(**{k: (getattr(self, k).to(dev) if getattr(self, k) is not None else None) for k in self.FIELDS})

    def repeat(self, c):  # (B,...) -> (B*c,...)  candidate expansion
        return ChainBatch(**{k: (getattr(self, k).repeat_interleave(c, 0) if getattr(self, k) is not None else None)
                             for k in self.FIELDS})

    @property
    def B(self): return self.origin.shape[0]
    @property
    def mid(self): return 0.5 * (self.lower + self.upper)
    @property
    def half(self): return (0.5 * (self.upper - self.lower)).clamp_min(1e-3)

    def q_to_u(self, q): return (q - self.mid) / self.half * self.mask
    def u_to_q(self, u): return (self.mid + self.half * u.clamp(-1, 1)) * self.mask
    def clamp_q(self, q): return torch.minimum(torch.maximum(q, self.lower), self.upper) * self.mask

    def sample_q(self, gen=None):
        r = torch.rand(self.lower.shape, device=self.lower.device, generator=gen)
        return (self.lower + r * (self.upper - self.lower)) * self.mask


def chains_to_batch(chains: List[KinChain], text_emb: Optional[np.ndarray] = None, nmax=NMAX) -> ChainBatch:
    M = len(chains)
    origin = np.tile(np.eye(4), (M, nmax, 1, 1)); axis = np.tile([0., 0., 1.], (M, nmax, 1))
    lo = np.zeros((M, nmax)); hi = np.zeros((M, nmax)); mask = np.zeros((M, nmax))
    tool = np.zeros((M, 4, 4)); reach = np.zeros(M)
    for m, c in enumerate(chains):
        n = c.dof
        assert n <= nmax, f"{c.name}: {n} DOF > NMAX={nmax}"
        origin[m, :n] = c.origins; axis[m, :n] = c.axes; lo[m, :n] = c.lower; hi[m, :n] = c.upper
        mask[m, :n] = 1; tool[m] = c.tool; reach[m] = c.reach
    f = lambda a: torch.tensor(a, dtype=torch.float32)
    return ChainBatch(origin=f(origin), axis=f(axis), lower=f(lo), upper=f(hi), mask=f(mask), tool=f(tool),
                      reach=f(reach), text=None if text_emb is None else f(text_emb), nidx=torch.arange(M))


def _skew(a):
    z = torch.zeros_like(a[..., 0])
    return torch.stack([z, -a[..., 2], a[..., 1], a[..., 2], z, -a[..., 0], -a[..., 1], a[..., 0], z], -1).view(*a.shape[:-1], 3, 3)


def axis_angle_R(axis, ang):
    K = _skew(axis); s = torch.sin(ang)[..., None, None]; c = torch.cos(ang)[..., None, None]
    I = torch.eye(3, device=axis.device).expand_as(K)
    return I + s * K + (1 - c) * (K @ K)


def fk(cb: ChainBatch, q):
    """Returns EE pose (B,4,4), joint positions P (B,N,3), world joint axes Z (B,N,3)."""
    B, N = q.shape
    T = torch.eye(4, device=q.device).expand(B, 4, 4)
    P, Z = [], []
    Rq = axis_angle_R(cb.axis, q * cb.mask)                       # (B,N,3,3)
    for i in range(N):
        Tj = T @ cb.origin[:, i]
        P.append(Tj[:, :3, 3]); Z.append((Tj[:, :3, :3] @ cb.axis[:, i, :, None])[..., 0])
        M = torch.zeros_like(Tj); M[:, :3, :3] = Rq[:, i]; M[:, 3, 3] = 1
        T = Tj @ M
    return T @ cb.tool, torch.stack(P, 1), torch.stack(Z, 1)


def rot6d(R): return torch.cat([R[..., :, 0], R[..., :, 1]], -1)


def rot_log(R):
    """SO(3) log map, robust near 0 and pi."""
    v = 0.5 * torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    c = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) * 0.5).clamp(-1, 1)
    s = v.norm(dim=-1)
    th = torch.atan2(s, c)
    out = v * (th / s.clamp_min(1e-7))[..., None]
    # near pi: axis from the symmetric part
    near_pi = (c < -0.99)
    if near_pi.any():
        # (R+R^T)/2 - cos(th) I = (1-cos th) a a^T
        Bm = (R + R.transpose(-1, -2)) * 0.5 - c[..., None, None] * torch.eye(3, device=R.device)
        d = torch.diagonal(Bm, dim1=-2, dim2=-1)
        k = d.argmax(-1, keepdim=True)          # column of R+R^T with largest norm ~ axis
        ref = torch.gather(Bm, -2, k[..., None].expand(*k.shape, 3)).squeeze(-2)
        ax = ref / ref.norm(dim=-1, keepdim=True).clamp_min(1e-7)
        ax = ax * torch.sign((ax * v).sum(-1, keepdim=True) + 1e-9)
        out = torch.where(near_pi[..., None], ax * th[..., None], out)
    return out


def pose_errors(Tee, goal):
    """(position error [m], rotation error [rad])."""
    pe = (Tee[:, :3, 3] - goal[:, :3, 3]).norm(dim=-1)
    return pe, rot_log(goal[:, :3, :3] @ Tee[:, :3, :3].transpose(-1, -2)).norm(dim=-1)


def pose_loss(Tee, goal, reach, w_rot=1.0):
    """Smooth, scale-free pose loss: |dp|/reach + ||R - Rg||_F / (2*sqrt2)  (~ theta/2)."""
    dp = ((Tee[:, :3, 3] - goal[:, :3, 3]).pow(2).sum(-1) + 1e-10).sqrt() / reach
    dr = ((Tee[:, :3, :3] - goal[:, :3, :3]).pow(2).sum((-1, -2)) + 1e-10).sqrt() / (2 * math.sqrt(2))
    return dp + w_rot * dr
