"""Download the journal_main checkpoint from the Hugging Face Hub and rebuild the network."""
import json
import os

import torch

from . import config
from . import kinematics
from .model import build_model


def fetch_checkpoint(repo=None, local_checkpoint=None):
    """Returns (model_path, training_config_dict_or_None).

    The checkpoint is cached by huggingface_hub (~/.cache/huggingface), so it downloads once.
    `local_checkpoint` is for offline development only."""
    if local_checkpoint:
        cfg_path = os.path.join(os.path.dirname(local_checkpoint), "config.json")
        return local_checkpoint, (json.load(open(cfg_path)) if os.path.exists(cfg_path) else None)
    from huggingface_hub import hf_hub_download
    repo = repo or config.DEFAULT_HF_REPO
    if repo.startswith("your-username"):
        raise SystemExit("Set the Hugging Face repo first: edit graphik_demo/config.py (DEFAULT_HF_REPO), "
                         "export GRAPHIK_HF_REPO=<user>/<repo>, or pass --repo <user>/<repo>.")
    token = os.environ.get("HF_TOKEN")            # only needed for a private repo (or use `hf auth login`)
    print(f"[hub] fetching {config.CHECKPOINT_FILE} from {repo} ...")
    model_path = hf_hub_download(repo, config.CHECKPOINT_FILE, token=token)
    try:
        cfg = json.load(open(hf_hub_download(repo, config.CHECKPOINT_CONFIG, token=token)))
    except Exception:
        cfg = None
    return model_path, cfg


def load_model(model_path, device="cpu"):
    """Rebuild the exact trained architecture from the hyper-parameters stored in the checkpoint."""
    kinematics.DEVICE = torch.device(device)
    ck = torch.load(model_path, map_location=device, weights_only=False)
    model = build_model(ck["cfg"], ck["text_dim"]).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] {ck['cfg'].get('arch', 'gnn')} | H={ck['cfg']['H']} layers={ck['cfg']['layers']} K={ck['cfg']['K']} "
          f"| text_dim={ck['text_dim']} | {n:.2f} M params | device={device}")
    return model, ck
