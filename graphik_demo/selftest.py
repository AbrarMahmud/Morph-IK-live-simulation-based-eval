"""Offline self-test (no network, no checkpoint): the whole pipeline with an untrained network of the same
architecture. Checks FK against PyBullet, IK convergence with the DLS polish, and headless rendering."""
import os
import tempfile

import numpy as np
import pybullet as p
import pybullet_data
import torch

from .kinematics import chains_to_batch, fk, parse_urdf_chain
from .model import build_model
from .robots import RobotEntry
from .session import IKSession


def run_selftest():
    ok = True
    path = os.path.join(pybullet_data.getDataPath(), "franka_panda/panda.urdf")
    chain = parse_urdf_chain(path, name="panda")
    # 1) FK equals PyBullet
    cid = p.connect(p.DIRECT)
    rid = p.loadURDF(path, useFixedBase=True, physicsClientId=cid)
    names = {p.getJointInfo(rid, i, physicsClientId=cid)[1].decode(): i for i in range(p.getNumJoints(rid, physicsClientId=cid))}
    ee = [i for i in range(p.getNumJoints(rid, physicsClientId=cid)) if p.getJointInfo(rid, i, physicsClientId=cid)[12].decode() == chain.ee_link][0]
    rng = np.random.default_rng(0); err = 0.0
    cb = chains_to_batch([chain])
    for _ in range(20):
        q = rng.uniform(chain.lower, chain.upper)
        for n, v in zip(chain.joint_names, q): p.resetJointState(rid, names[n], v, physicsClientId=cid)
        ls = p.getLinkState(rid, ee, computeForwardKinematics=True, physicsClientId=cid)
        qt = torch.zeros(1, 8); qt[0, :7] = torch.tensor(q, dtype=torch.float32)
        err = max(err, float(np.linalg.norm(fk(cb, qt)[0][0, :3, 3].numpy() - np.array(ls[4]))))
    p.disconnect(cid)
    print(f"[selftest] FK vs PyBullet max error: {err:.2e} m", "OK" if err < 1e-5 else "FAIL"); ok &= err < 1e-5
    # 2) solver converges (untrained net + DLS polish)
    torch.manual_seed(0)
    model = build_model(dict(H=32, layers=1, heads=4, K=4, arch="gnn"), 0).eval()
    s = IKSession(model); s.set_robot(chain)
    succ = 0
    for _ in range(20):
        T = chain.fk_np(rng.uniform(chain.lower, chain.upper))
        s.reset(rng.uniform(chain.lower, chain.upper))
        _, info = s.solve(T, use_seed=True, polish=True, polish_iters=40)
        succ += info["pos_err_mm"] < 5 and info["rot_err_deg"] < 2
    print(f"[selftest] IK pipeline (untrained net + polish) reached {succ}/20 targets", "OK" if succ >= 12 else "FAIL")
    ok &= succ >= 12
    # 3) headless rendering + video writer
    from .record import record
    out = os.path.join(tempfile.mkdtemp(), "selftest.mp4")
    r = record(s, RobotEntry("panda", chain, "SELFTEST"), "circle", out, frames=12, res=256)
    good = os.path.exists(r["video"]) and os.path.getsize(r["video"]) > 1000
    print(f"[selftest] headless render -> {r['video']}", "OK" if good else "FAIL"); ok &= good
    print("[selftest]", "ALL OK" if ok else "FAILED")
    return 0 if ok else 1
