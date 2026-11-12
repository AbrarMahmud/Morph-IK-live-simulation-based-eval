"""Interactive PyBullet window for reviewers.

Right-hand panel (PyBullet "Params"):
  buttons : previous / next robot, previous / next track, random far target (solved with NO seed),
            apply / reset link lengths
  sliders : speed, track size, seed-with-current-joints (0/1), DLS polish (0/1), pause (0/1),
            manual target dx/dy/dz/roll/pitch/yaw, per-link length multipliers
3-D view : red = target track, green = achieved end-effector trail, RGB axes = target frame,
           HUD text = robot, SEEN/UNSEEN status, track, live position / rotation error and solve time.
"""
import math
import time

import numpy as np
import pybullet as p
import pybullet_data

from .render import load_body, set_joints
from .trajectories import TRACK_HELP, TRACKS, build_track, manual_goal


class Button:
    def __init__(self, name):
        self.id = p.addUserDebugParameter(name, 1, 0, 0); self.n = 0

    def pressed(self):
        v = p.readUserDebugParameter(self.id)
        hit = v > self.n; self.n = v
        return hit


class DemoGUI:
    def __init__(self, session, robots, robot=None, track="figure8", fps=30, width=1400, height=900):
        self.s, self.robots, self.fps = session, robots, fps
        self.ri = next((i for i, r in enumerate(robots) if r.name == robot), 0)
        self.ti = TRACKS.index(track)
        self.cid = p.connect(p.GUI, options=f"--width={width} --height={height}")
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        for flag in (p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                     p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW):
            p.configureDebugVisualizer(flag, 0)                     # hide the empty camera-preview panels
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.scales = None; self.track_seed = 0; self.hud = {}; self.status_t = 0
        self.load_robot()

    # ------------------------------------------------------------------ setup
    def entry(self):
        return self.robots[self.ri]

    def load_robot(self):
        p.resetSimulation(); p.removeAllUserDebugItems(); self.hud = {}
        e = self.entry()
        e.link_scales = self.scales
        self.chain = e.scaled_chain()
        self.variant = self.chain is not e.chain
        p.loadURDF("plane.urdf", [0, 0, -0.001])
        self.rid, self.jidx, self.render_kind = load_body(self.chain, self.cid, prefer_meshes=not self.variant)
        self.s.set_robot(self.chain)
        self.build_params()
        self.load_track()

    def build_params(self):
        p.removeAllUserParameters()
        self.b_prev_robot, self.b_next_robot = Button("<< previous robot"), Button("next robot >>")
        self.b_prev_track, self.b_next_track = Button("<< previous track"), Button("next track >>")
        self.b_new_track = Button("new track position")
        self.b_far = Button("random far target (solved with NO seed)")
        self.sl_speed = p.addUserDebugParameter("speed", 0.1, 3.0, 1.0)
        self.sl_size = p.addUserDebugParameter("track size", 0.3, 2.5, 1.0); self.size = 1.0; self.size_t = None
        self.sl_seed = p.addUserDebugParameter("seed with current joints (1=yes 0=no)", 0, 1, 1)
        self.sl_polish = p.addUserDebugParameter("DLS polish (1=on, 0=network only)", 0, 1, 1)
        self.sl_pause = p.addUserDebugParameter("pause (1=yes)", 0, 1, 0)
        R = math.pi
        self.sl_manual = [p.addUserDebugParameter(f"manual target {n}", lo, hi, 0.0) for n, lo, hi in
                          (("dx (x reach)", -0.4, 0.4), ("dy (x reach)", -0.4, 0.4), ("dz (x reach)", -0.4, 0.4),
                           ("roll", -R, R), ("pitch", -R, R), ("yaw", -R, R))]
        e = self.entry()
        lengths = [np.linalg.norm(e.chain.origins[i, :3, 3]) for i in range(e.chain.dof)] + [np.linalg.norm(e.chain.tool[:3, 3])]
        self.link_sliders = []
        cur = self.scales if self.scales is not None else np.ones(e.n_links)
        for i, L in enumerate(lengths):
            if L < 1e-3:
                self.link_sliders.append(None); continue
            nm = f"link {i} length x  ({'tool' if i == e.chain.dof else f'to joint {i + 1}'}, {L:.3f} m)"
            self.link_sliders.append(p.addUserDebugParameter(nm, 0.5, 1.8, float(cur[i])))
        self.b_apply_links, self.b_reset_links = Button("apply link lengths (UNSEEN variant)"), Button("reset link lengths")

    def load_track(self):
        kind = TRACKS[self.ti]
        self.q0, self.goals, self.note = build_track(self.chain, kind, n=240, size=self.size, seed=self.track_seed)
        self.s.reset(self.q0); set_joints(self.rid, self.jidx, self.q0, self.cid)
        self.phase, self.prev_ee, self.far_goal = 0.0, None, None
        p.removeAllUserDebugItems(); self.hud = {}
        if kind not in ("manual", "random_poses"):
            pts = [g[:3, 3] for g in self.goals] + [self.goals[0][:3, 3]]
            for a, b in zip(pts[:-1], pts[1:]):
                p.addUserDebugLine(a, b, [0.9, 0.1, 0.1], 2.0)
        c = np.mean([g[:3, 3] for g in self.goals], 0); R = self.chain.reach
        self.cam_target = np.array([0.5 * c[0], 0.5 * c[1], 0.6 * c[2]])
        p.resetDebugVisualizerCamera(cameraDistance=1.25 * R, cameraYaw=50, cameraPitch=-20,
                                     cameraTargetPosition=self.cam_target.tolist())
        hor = np.array(p.getDebugVisualizerCamera()[6]); hor[2] = 0
        self.cam_left = -hor / max(np.linalg.norm(hor), 1e-9)      # HUD is written from the left side of the view
        print(f"[track] {self.entry().name}: {self.note}")

    # ------------------------------------------------------------------ drawing
    def text(self, key, msg, row, color=(0, 0, 0)):
        R = self.chain.reach                                   # HUD rows just below the top of the view
        pos = (self.cam_target + 0.8 * R * self.cam_left + np.array([0, 0, (0.56 - 0.065 * row) * R])).tolist()
        kw = dict(replaceItemUniqueId=self.hud[key]) if key in self.hud else {}
        self.hud[key] = p.addUserDebugText(msg, pos, color, textSize=1.3, **kw)

    def frame_axes(self, T):
        L = 0.08 * self.chain.reach
        for i, c in enumerate(([1, 0, 0], [0, 1, 0], [0, 0, 1])):
            key = f"ax{i}"
            kw = dict(replaceItemUniqueId=self.hud[key]) if key in self.hud else {}
            self.hud[key] = p.addUserDebugLine(T[:3, 3], T[:3, 3] + L * T[:3, i], c, 3.0, **kw)

    # ------------------------------------------------------------------ loop
    def handle_inputs(self):
        n = len(self.robots)
        if self.b_next_robot.pressed(): self.ri = (self.ri + 1) % n; self.scales = None; self.load_robot(); return True
        if self.b_prev_robot.pressed(): self.ri = (self.ri - 1) % n; self.scales = None; self.load_robot(); return True
        if self.b_next_track.pressed(): self.ti = (self.ti + 1) % len(TRACKS); self.load_track()
        if self.b_prev_track.pressed(): self.ti = (self.ti - 1) % len(TRACKS); self.load_track()
        if self.b_new_track.pressed(): self.track_seed += 1; self.load_track()
        if self.b_apply_links.pressed():
            self.scales = np.array([1.0 if sid is None else p.readUserDebugParameter(sid) for sid in self.link_sliders])
            self.load_robot(); return True
        if self.b_reset_links.pressed(): self.scales = None; self.load_robot(); return True
        if self.b_far.pressed():
            q = np.random.default_rng().uniform(self.chain.lower, self.chain.upper)
            self.far_goal = (self.chain.fk_np(q), time.time() + 2.5)
        size = p.readUserDebugParameter(self.sl_size)
        if abs(size - self.size) > 1e-3:
            self.size, self.size_t = size, time.time()
        if self.size_t and time.time() - self.size_t > 0.4:       # debounce slider drags
            self.size_t = None; self.load_track()
        return False

    def current_goal(self):
        if self.far_goal is not None:
            if time.time() < self.far_goal[1]:
                return self.far_goal[0], False                       # far target: solve WITHOUT seed
            self.far_goal = None
        kind = TRACKS[self.ti]
        if kind == "manual":
            v = [p.readUserDebugParameter(i) for i in self.sl_manual]
            return manual_goal(self.goals[0], *v, self.chain.reach), None
        if p.readUserDebugParameter(self.sl_pause) < 0.5:
            self.phase += p.readUserDebugParameter(self.sl_speed)
        return self.goals[int(self.phase) % len(self.goals)], None

    def run(self, max_frames=None):
        dt = 1.0 / self.fps
        frame = 0
        while p.isConnected(self.cid) and (max_frames is None or frame < max_frames):
            frame += 1
            t0 = time.time()
            try:
                if self.handle_inputs():
                    continue
                goal, force_seed = self.current_goal()
                use_seed = p.readUserDebugParameter(self.sl_seed) > 0.5 if force_seed is None else force_seed
                polish = p.readUserDebugParameter(self.sl_polish) > 0.5
                q, info = self.s.solve(goal, use_seed=use_seed, polish=polish)
                set_joints(self.rid, self.jidx, q, self.cid)
                ee = info["ee"][:3, 3]
                if self.prev_ee is not None and np.linalg.norm(ee - self.prev_ee) < 0.2 * self.chain.reach:
                    p.addUserDebugLine(self.prev_ee, ee, [0.1, 0.75, 0.2], 3.0, lifeTime=8)
                self.prev_ee = ee
                self.frame_axes(goal)
                e = self.entry()
                status = "UNSEEN VARIANT (link lengths changed)" if self.variant else e.status
                ok = info["pos_err_mm"] < 5 and info["rot_err_deg"] < 2
                self.text("l0", f"{e.name}  [{self.chain.dof}-DOF, reach {self.chain.reach:.2f} m]  -  {status}", 0,
                          (0.0, 0.45, 0.0) if status == "SEEN" else (0.75, 0.25, 0.0))
                self.text("l1", f"track: {TRACKS[self.ti]} - {TRACK_HELP[TRACKS[self.ti]]}", 1)
                self.text("l2", f"seed: {'current joints' if use_seed else 'NONE (from scratch)'}  |  "
                                f"{'GNN + DLS polish' if polish else 'GNN only'}  |  render: {self.render_kind}", 2)
                self.text("l3", f"pos err {info['pos_err_mm']:.3f} mm   rot err {info['rot_err_deg']:.3f} deg   "
                                f"joint step {info['jump_deg']:.1f} deg   solve {info['ms']:.0f} ms"
                                + ("" if ok else "   (target outside workspace?)"), 3,
                          (0, 0, 0) if ok else (0.8, 0, 0))
            except p.error:
                break
            time.sleep(max(0.0, dt - (time.time() - t0)))
