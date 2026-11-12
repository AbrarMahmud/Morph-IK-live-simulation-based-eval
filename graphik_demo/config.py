"""Demo configuration. Edit DEFAULT_HF_REPO once before publishing (or set GRAPHIK_HF_REPO / --repo)."""
import os

# Hugging Face repo that the training notebook pushed to.
DEFAULT_HF_REPO = os.environ.get("GRAPHIK_HF_REPO", "iFaz/urdf-graph-ik-journal-main-seed_1")

# The ONLY checkpoint this demo uses: journal profile, main experiment, seed 0.
CHECKPOINT_EXPERIMENT = "main_s1"
CHECKPOINT_FILE = f"experiments/{CHECKPOINT_EXPERIMENT}/model.pt"
CHECKPOINT_CONFIG = f"experiments/{CHECKPOINT_EXPERIMENT}/config.json"

# Same sentence encoder as in training (only used if the checkpoint was trained with text).
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROBOTS_YAML = os.path.join(REPO_ROOT, "robots", "robots.yaml")
