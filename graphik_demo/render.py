"""Load a robot into PyBullet for display: real meshes when possible, else a stick model generated from the
kinematic chain (always used for link-length variants, whose meshes would not match)."""
import os

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
