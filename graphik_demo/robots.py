"""Robot registry: robots/robots.yaml + SEEN / UNSEEN status from the checkpoint's training config."""
import copy
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import yaml

from . import config
from .kinematics import KinChain, parse_urdf_chain


def resolve_path(urdf):
    if urdf.startswith("pybullet_data://"):
        import pybullet_data
        return os.path.join(pybullet_data.getDataPath(), urdf[len("pybullet_data://"):])
    return urdf if os.path.isabs(urdf) else os.path.join(config.REPO_ROOT, urdf)


@dataclass
class RobotEntry:
    name: str
    chain: KinChain
    status: str                          # "SEEN", "UNSEEN (held out)", "UNSEEN (new robot)"
    link_scales: Optional[np.ndarray] = field(default=None)

    def scaled_chain(self, scales=None):
        """Same robot with each link (joint-to-joint offset) stretched: the 'changed extrusion' case."""
        scales = self.link_scales if scales is None else np.asarray(scales, dtype=float)
        if scales is None or np.allclose(scales, 1.0):
            return self.chain
        c = copy.deepcopy(self.chain)
        for i in range(c.dof):
            c.origins[i, :3, 3] = self.chain.origins[i, :3, 3] * scales[i]
        c.tool[:3, 3] = self.chain.tool[:3, 3] * scales[-1]
        c.name = f"{self.name}_variant"; c.source = ""
        return c

    @property
    def n_links(self):
        return self.chain.dof + 1           # dof joint offsets + tool offset


def status_from_training_config(name, train_cfg):
    if not train_cfg:
        return "UNSEEN? (no training config)"
    if name in train_cfg.get("train", []):
        return "SEEN"
    if name in train_cfg.get("unseen", []):
        return "UNSEEN (held out)"
    return "UNSEEN (new robot)"


def load_robots(train_cfg=None, yaml_path=None, extra_urdf=None, extra_ee=None, verbose=True) -> List[RobotEntry]:
    spec = yaml.safe_load(open(yaml_path or config.ROBOTS_YAML))["robots"]
    if extra_urdf:
        spec = [dict(name=os.path.splitext(os.path.basename(extra_urdf))[0], urdf=os.path.abspath(extra_urdf), ee=extra_ee)] + spec
    out, skipped = [], []
    for s in spec:
        path = resolve_path(s["urdf"])
        if not os.path.exists(path):
            skipped.append(s["name"]); continue
        try:
            ch = parse_urdf_chain(path, ee_link=s.get("ee"), exclude=s.get("exclude", ()), name=s["name"])
        except Exception as e:
            print(f"[robots] {s['name']}: cannot parse ({e})"); continue
        out.append(RobotEntry(s["name"], ch, status_from_training_config(s["name"], train_cfg)))
    if verbose:
        for r in out:
            print(f"  {r.name:18s} {r.chain.dof}-DOF  reach {r.chain.reach:.2f} m  ee={r.chain.ee_link:16s} {r.status}")
        if skipped:
            print(f"  (not found, skipped: {', '.join(skipped)} -> see README 'Industrial robots')")
    if not out:
        raise SystemExit("No robot URDF found. Check robots/robots.yaml.")
    return out
