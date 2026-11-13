"""Headless mode: run the same live IK loop without a window and write an mp4 (or GIF) with an overlay.
Useful on servers / CI or to attach videos to a review."""
import os

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont

from .render import LongExposure, load_body, set_joints
from .trajectories import build_track


def _font(size=14):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                    # Pillow < 10.1
        return ImageFont.load_default()


def record(session, entry, track="figure8", out="videos/demo.mp4", frames=240, size=1.0, scales=None,
           use_seed=True, polish=True, fps=30, res=640, seed=0, exposure=0, exposure_every=5):
    """exposure > 0 keeps that many translucent copies of earlier steps (long-exposure look) and also saves the
    final frame as <out>_exposure.png (a still for papers / slides)."""
    entry.link_scales = None if scales is None else np.asarray(scales, dtype=float)
    chain = entry.scaled_chain()
    variant = chain is not entry.chain
    cid = p.connect(p.DIRECT)
    try:
        rid, jidx, kind = load_body(chain, cid, prefer_meshes=not variant)
        session.set_robot(chain)
        q0, goals, note = build_track(chain, track, n=frames, size=size, seed=seed)
        session.reset(q0); set_joints(rid, jidx, q0, cid)
        ghosts = LongExposure(chain, cid, prefer_meshes=kind == "meshes", count=exposure, every=exposure_every) if exposure else None
        R = chain.reach
        dot = p.createVisualShape(p.GEOM_SPHERE, radius=0.008 * R, rgbaColor=[1, 0.1, 0.1, 0.8], physicsClientId=cid)
        trail = p.createVisualShape(p.GEOM_SPHERE, radius=0.006 * R, rgbaColor=[0.1, 0.8, 0.2, 1], physicsClientId=cid)
        if track not in ("manual", "random_poses"):
            for T in goals[::3]:
                p.createMultiBody(baseVisualShapeIndex=dot, basePosition=T[:3, 3].tolist(), physicsClientId=cid)
        center = np.mean([g[:3, 3] for g in goals], 0)
        view = p.computeViewMatrixFromYawPitchRoll([0.5 * center[0], 0.5 * center[1], 0.6 * center[2]], 1.2 * R,
                                                   50, -22, 0, 2, physicsClientId=cid)
        proj = p.computeProjectionMatrixFOV(50, 1.0, 0.01, 50 * R, physicsClientId=cid)
        status = "UNSEEN VARIANT (link lengths changed)" if variant else entry.status
        font, frames_out, rows = _font(15), [], []
        for k in range(frames):
            goal = goals[k % len(goals)]
            q, info = session.solve(goal, use_seed=use_seed, polish=polish)
            set_joints(rid, jidx, q, cid); rows.append(info)
            if ghosts is not None:
                ghosts.push(q)
            if k % 2 == 0:
                p.createMultiBody(baseVisualShapeIndex=trail, basePosition=info["ee"][:3, 3].tolist(), physicsClientId=cid)
            img = p.getCameraImage(res, res, view, proj, renderer=p.ER_TINY_RENDERER, physicsClientId=cid)[2]
            im = Image.fromarray(np.reshape(np.asarray(img, dtype=np.uint8), (res, res, 4))[..., :3])
            d = ImageDraw.Draw(im)
            d.rectangle([0, 0, res, 66], fill=(255, 255, 255))
            d.text((8, 4), f"{entry.name} ({chain.dof}-DOF)  -  {status}", fill=(0, 110, 0) if status == "SEEN" else (190, 70, 0), font=font)
            d.text((8, 24), f"track: {note}  |  seed: {'current joints' if use_seed else 'none'}  |  "
                            f"{'GNN+DLS' if polish else 'GNN only'}", fill=(0, 0, 0), font=font)
            d.text((8, 44), f"frame {k + 1}/{frames}  pos err {info['pos_err_mm']:.3f} mm  rot err {info['rot_err_deg']:.3f} deg  "
                            f"joint step {info['jump_deg']:.1f} deg  solve {info['ms']:.0f} ms", fill=(0, 0, 0), font=font)
            frames_out.append(np.asarray(im))
        still = None
        if ghosts is not None and frames_out:
            still = os.path.splitext(out)[0] + "_exposure.png"
            os.makedirs(os.path.dirname(os.path.abspath(still)), exist_ok=True)
            hi = 2 * res                                     # high-resolution still without the text overlay
            img = p.getCameraImage(hi, hi, view, proj, renderer=p.ER_TINY_RENDERER, physicsClientId=cid)[2]
            Image.fromarray(np.reshape(np.asarray(img, dtype=np.uint8), (hi, hi, 4))[..., :3]).save(still)
    finally:
        p.disconnect(cid)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    import imageio
    try:
        imageio.mimwrite(out, frames_out, fps=fps, codec="libx264", quality=7, macro_block_size=1)
    except Exception:
        out = os.path.splitext(out)[0] + ".gif"
        imageio.mimwrite(out, frames_out, duration=1 / fps, loop=0)
    pe = np.array([r["pos_err_mm"] for r in rows]); re = np.array([r["rot_err_deg"] for r in rows])
    js = np.array([r["jump_deg"] for r in rows[1:]]) if len(rows) > 1 else np.zeros(1)
    ms = np.array([r["ms"] for r in rows])
    summary = dict(robot=entry.name, status=status, track=note, render=kind, pos_mm_mean=pe.mean(), pos_mm_max=pe.max(),
                   rot_deg_max=re.max(), frac_ok=float(((pe < 5) & (re < 2)).mean()), max_joint_step_deg=js.max(),
                   solve_ms_median=float(np.median(ms)), video=out)
    if still:
        summary["exposure_still"] = still
    return summary
