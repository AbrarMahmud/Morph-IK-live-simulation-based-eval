"""Offline tests (no download): run with `pytest -q`."""
import os

import numpy as np
import pybullet_data
import torch

from graphik_demo.kinematics import chains_to_batch, fk, parse_urdf_chain
from graphik_demo.model import build_model
from graphik_demo.robots import RobotEntry, status_from_training_config
from graphik_demo.session import IKSession
from graphik_demo.trajectories import TRACKS, build_track

PANDA = os.path.join(pybullet_data.getDataPath(), "franka_panda/panda.urdf")


def test_selftest_passes():
    from graphik_demo.selftest import run_selftest
    assert run_selftest() == 0


def test_all_tracks_build_and_are_solved():
    ch = parse_urdf_chain(PANDA, name="panda")
    torch.manual_seed(0)
    s = IKSession(build_model(dict(H=32, layers=1, heads=4, K=4), 0).eval()); s.set_robot(ch)
    for t in TRACKS:
        q0, goals, note = build_track(ch, t, n=24)
        assert len(goals) >= 1
        s.reset(q0)
        _, info = s.solve(goals[min(1, len(goals) - 1)], polish_iters=30)
        assert np.isfinite(info["pos_err_mm"])


def test_link_scaling_changes_kinematics():
    ch = parse_urdf_chain(PANDA, name="panda")
    e = RobotEntry("panda", ch, "SEEN")
    v = e.scaled_chain(np.full(e.n_links, 1.3))
    q = torch.zeros(1, 8)
    p0 = fk(chains_to_batch([ch]), q)[0][0, :3, 3]
    p1 = fk(chains_to_batch([v]), q)[0][0, :3, 3]
    assert torch.allclose(p1, 1.3 * p0, atol=1e-5)


def test_status_labels():
    cfg = {"train": ["xarm6"], "unseen": ["panda"]}
    assert status_from_training_config("xarm6", cfg) == "SEEN"
    assert status_from_training_config("panda", cfg) == "UNSEEN (held out)"
    assert status_from_training_config("ur5", cfg) == "UNSEEN (new robot)"
