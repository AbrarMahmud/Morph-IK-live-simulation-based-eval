"""Load a robot into PyBullet for display: real meshes when possible, else a stick model generated from the
kinematic chain (always used for link-length variants, whose meshes would not match)."""
import os
from contextlib import contextmanager

import numpy as np
import pybullet as p
import pybullet_data

from .kinematics import KinChain
from .urdf_gen import chain_to_urdf


def load_body(chain: KinChain, cid, prefer_meshes=True):
    """Returns (body_id, joint_indices_in_chain_order, 'meshes' | 'stick')."""
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
    if prefer_meshes and chain.source and os.path.exists(chain.source):
        try:
            rid = p.loadURDF(chain.source, useFixedBase=True, physicsClientId=cid,
                             flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
            names = {p.getJointInfo(rid, i, physicsClientId=cid)[1].decode(): i
                     for i in range(p.getNumJoints(rid, physicsClientId=cid))}
            has_visuals = len(p.getVisualShapeData(rid, physicsClientId=cid)) > 0
            if has_visuals and all(n in names for n in chain.joint_names):
                return rid, [names[n] for n in chain.joint_names], "meshes"
            p.removeBody(rid, physicsClientId=cid)
        except Exception:
            pass
    rid = p.loadURDF(chain_to_urdf(chain), useFixedBase=True, physicsClientId=cid)
    return rid, list(range(chain.dof)), "stick"


def set_joints(rid, jidx, q, cid):
    for j, v in zip(jidx, q):
        p.resetJointState(rid, j, float(v), physicsClientId=cid)


@contextmanager
def batched(cid):
    """Pause drawing while issuing many visual changes. In the GUI each changeVisualShape otherwise waits for a
    redraw (~1 frame per call); batched, a whole re-colour costs well under a millisecond per call."""
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=cid)
    try:
        yield
    finally:
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=cid)


class LongExposure:
    """Long-exposure / stroboscopic effect: translucent copies ("ghosts") of the robot at earlier steps.

    Every `every`-th call to push(q) stamps a ghost at q (only if the arm actually moved). At most `count`
    ghosts are kept; the oldest is re-used for the newest stamp, and ghosts fade from faint/light (old) to
    strong/dark (recent), so the current, fully opaque robot reads as the end of the motion. Ghost bodies are
    created one per stamp (not all at once), so switching it on never freezes the window."""

    OLD, NEW = np.array([0.80, 0.86, 1.00]), np.array([0.15, 0.35, 0.90])     # light -> dark blue

    def __init__(self, chain: KinChain, cid, prefer_meshes=True, count=12, every=5, alpha=(0.12, 0.55)):
        self.chain, self.cid, self.prefer_meshes = chain, cid, prefer_meshes
        self.count, self.every, self.alpha = int(count), max(1, int(every)), alpha
        self.ghosts = []                    # [(body_id, joint_indices, [link indices])], oldest first
        self.calls, self.last_q = 0, None

    def _new_ghost(self):
        rid, jidx, _ = load_body(self.chain, self.cid, prefer_meshes=self.prefer_meshes)
        links = [-1] + list(range(p.getNumJoints(rid, physicsClientId=self.cid)))
        for l in links:                     # visual only: never collides with / disturbs the live robot
            p.setCollisionFilterGroupMask(rid, l, 0, 0, physicsClientId=self.cid)
        return rid, jidx, links

    def _recolor(self):
        k = len(self.ghosts)
        for i, (rid, _, links) in enumerate(self.ghosts):
            t = (i + 1) / k                                       # 0 = oldest ... 1 = newest
            rgb = (1 - t) * self.OLD + t * self.NEW
            a = self.alpha[0] + (self.alpha[1] - self.alpha[0]) * t
            for l in links:                 # the base never moves: drawing it again would only add clutter
                p.changeVisualShape(rid, l, rgbaColor=[*rgb, 0.0 if l == -1 else a], physicsClientId=self.cid)

    def configure(self, count=None, every=None):
        if every is not None:
            self.every = max(1, int(every))
        if count is not None and max(1, int(count)) != self.count:
            self.count = max(1, int(count))
            with batched(self.cid):
                while len(self.ghosts) > self.count:              # drop the oldest
                    p.removeBody(self.ghosts.pop(0)[0], physicsClientId=self.cid)
                if self.ghosts:
                    self._recolor()

    def push(self, q, force=False):
        q = np.asarray(q, dtype=float)
        self.calls += 1
        if not force and self.calls % self.every:
            return False
        if self.last_q is not None and np.max(np.abs(q - self.last_q)) < 1e-3:
            return False                                          # paused / not moving: no duplicate ghosts
        with batched(self.cid):
            g = self.ghosts.pop(0) if len(self.ghosts) >= self.count else self._new_ghost()
            set_joints(g[0], g[1], q, self.cid)
            self.ghosts.append(g); self.last_q = q.copy()
            self._recolor()
        return True

    def clear(self):
        try:
            with batched(self.cid):
                for rid, _, _ in self.ghosts:
                    p.removeBody(rid, physicsClientId=self.cid)
        except p.error:                                           # window already closed
            pass
        self.ghosts, self.calls, self.last_q = [], 0, None
