"""Write a kinematically identical URDF (links drawn as cylinders) for any KinChain.

Generated from the training notebook (ik-v8-urdf-graph-ik.ipynb) so the demo runs exactly the
trained architecture. Do not edit by hand; regenerate instead."""
import os, tempfile
import numpy as np

from .kinematics import KinChain, R_to_rpy


def _rpy_str(R): return " ".join(f"{v:.10g}" for v in R_to_rpy(R))


def _xyz_str(t): return " ".join(f"{v:.10g}" for v in t)


def _align_z(v):
    """Rotation whose z axis points along v (for drawing a link as a cylinder)."""
    n = np.linalg.norm(v)
    if n < 1e-9: return np.eye(3)
    z = v / n
    a = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(a, z); x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], 1)


def _xml(text):
    from xml.sax.saxutils import escape
    return escape(str(text), {'"': "&quot;"})


def chain_to_urdf(ch: KinChain, path=None, radius=None):
    """Write a kinematically identical URDF (links drawn as cylinders) for PyBullet / ikpy / TRAC-IK."""
    r = radius or 0.02 * ch.reach
    def link(name, t_next):
        L = np.linalg.norm(t_next)
        vis = ""
        if L > 1e-6:
            vis = (f'<visual><origin xyz="{_xyz_str(t_next / 2)}" rpy="{_rpy_str(_align_z(t_next))}"/>'
                   f'<geometry><cylinder radius="{r:.5g}" length="{L:.6g}"/></geometry>'
                   f'<material name="grey"><color rgba="0.75 0.75 0.8 1"/></material></visual>')
        joint_ball = (f'<visual><geometry><sphere radius="{1.4 * r:.5g}"/></geometry>'
                      f'<material name="orange"><color rgba="1 0.55 0.1 1"/></material></visual>')
        return (f'<link name="{name}"><inertial><mass value="0.1"/><inertia ixx="1e-4" iyy="1e-4" izz="1e-4" '
                f'ixy="0" ixz="0" iyz="0"/></inertial>{vis}{joint_ball}</link>')
    # link l_i holds the cylinder to the NEXT joint origin (expressed in l_i's frame)
    parts = [f'<robot name="{_xml(ch.name)}">', link("l0", ch.origins[0, :3, 3])]
    for i in range(ch.dof):
        t_next = ch.origins[i + 1, :3, 3] if i + 1 < ch.dof else ch.tool[:3, 3]
        parts.append(link(f"l{i + 1}", t_next))
        O = ch.origins[i]
        parts.append(f'<joint name="{_xml(ch.joint_names[i])}" type="revolute"><parent link="l{i}"/><child link="l{i + 1}"/>'
                     f'<origin xyz="{_xyz_str(O[:3, 3])}" rpy="{_rpy_str(O[:3, :3])}"/><axis xyz="{_xyz_str(ch.axes[i])}"/>'
                     f'<limit lower="{ch.lower[i]:.10g}" upper="{ch.upper[i]:.10g}" effort="100" velocity="3"/></joint>')
    parts.append('<link name="ee"><inertial><mass value="0.01"/><inertia ixx="1e-5" iyy="1e-5" izz="1e-5" ixy="0" ixz="0" iyz="0"/></inertial>'
                 f'<visual><geometry><sphere radius="{1.6 * r:.5g}"/></geometry><material name="green"><color rgba="0.1 0.8 0.2 1"/></material></visual></link>')
    parts.append(f'<joint name="tool_joint" type="fixed"><parent link="l{ch.dof}"/><child link="ee"/>'
                 f'<origin xyz="{_xyz_str(ch.tool[:3, 3])}" rpy="{_rpy_str(ch.tool[:3, :3])}"/></joint></robot>')
    if path is None:
        safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in ch.name) or "robot"   # valid on Windows too
        path = os.path.join(tempfile.mkdtemp(prefix="urdf_"), f"{safe}.urdf")
    with open(path, "w") as f: f.write("\n".join(parts))
    return path
