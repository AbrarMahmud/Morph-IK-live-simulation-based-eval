"""Command line:

  python -m graphik_demo                       live simulation window (default)
  python -m graphik_demo gui --robot panda --track circle
  python -m graphik_demo record --robot xarm6 --track figure8 --out videos/xarm6.mp4
  python -m graphik_demo record --robot xarm6 --link-scale 1,1.3,1.3,1,1,1,1     # stretched (unseen) variant
  python -m graphik_demo record --robot panda --track circle --exposure 14   # long exposure (+ still .png)
  python -m graphik_demo list                  robots and their SEEN / UNSEEN status
  python -m graphik_demo selftest              offline check of the whole pipeline (no download needed)
"""
import argparse
import json
import os

import torch

from . import config


def _model_and_robots(args):
    from .hub import fetch_checkpoint, load_model
    from .robots import load_robots
    from .text import make_text_encoder
    path, train_cfg = fetch_checkpoint(args.repo, args.checkpoint)
    model, ck = load_model(path)
    if train_cfg:
        print(f"[checkpoint] trained on {train_cfg.get('train')} | held out {train_cfg.get('unseen')}")
    text = make_text_encoder(enabled=(ck["text_dim"] > 0 and not args.no_text))
    robots = load_robots(train_cfg, extra_urdf=args.urdf, extra_ee=args.ee)
    return model, text, robots


def main(argv=None):
    from .trajectories import TRACKS
    ap = argparse.ArgumentParser(prog="graphik_demo", description="Live demo of the URDF-conditioned graph IK model "
                                 f"(checkpoint: {config.CHECKPOINT_EXPERIMENT})")
    ap.add_argument("command", nargs="?", default="gui", choices=["gui", "record", "list", "selftest"])
    ap.add_argument("--repo", help=f"Hugging Face repo (default: {config.DEFAULT_HF_REPO})")
    ap.add_argument("--checkpoint", help="local model.pt instead of downloading (offline development)")
    ap.add_argument("--robot", help="robot name from robots/robots.yaml (see `list`)")
    ap.add_argument("--urdf", help="add any other URDF file (shown as an UNSEEN robot)")
    ap.add_argument("--ee", help="end-effector link name for --urdf")
    ap.add_argument("--track", default="figure8", choices=TRACKS)
    ap.add_argument("--size", type=float, default=1.0, help="track size multiplier")
    ap.add_argument("--link-scale", help="comma-separated length multipliers, one per joint offset + tool")
    ap.add_argument("--no-seed", action="store_true", help="never use the current joints as a seed")
    ap.add_argument("--no-polish", action="store_true", help="network output only (no DLS polish)")
    ap.add_argument("--no-text", action="store_true", help="skip the optional text encoder")
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--exposure", type=int, default=0, help="record: keep N translucent copies of earlier steps "
                    "(long-exposure look) and save the last frame as a .png still")
    ap.add_argument("--exposure-every", type=int, default=5, help="record: frames between those copies")
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)

    if args.command == "selftest":
        from .selftest import run_selftest
        return run_selftest()

    model, text, robots = _model_and_robots(args)
    if args.command == "list":
        return 0
    from .session import IKSession
    session = IKSession(model, text)
    if args.command == "gui":
        from .app_gui import DemoGUI
        DemoGUI(session, robots, robot=args.robot or (args.urdf and robots[0].name), track=args.track).run()
        return 0
    # record
    names = [r.name for r in robots]
    targets = [args.robot] if args.robot else names
    results = []
    from .record import record
    for name in targets:
        if name not in names:
            raise SystemExit(f"unknown robot {name}; available: {names}")
        entry = robots[names.index(name)]
        scales = [float(v) for v in args.link_scale.split(",")] if args.link_scale else None
        if scales is not None and len(scales) != entry.n_links:
            raise SystemExit(f"{name} needs {entry.n_links} link scales (one per joint offset + tool), got {len(scales)}")
        out = args.out if (args.out and len(targets) == 1) else f"videos/{name}_{args.track}.mp4"
        r = record(session, entry, args.track, out, args.frames, args.size, scales,
                   use_seed=not args.no_seed, polish=not args.no_polish,
                   exposure=args.exposure, exposure_every=args.exposure_every)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))
        results.append(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
