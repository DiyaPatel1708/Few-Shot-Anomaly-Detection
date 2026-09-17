import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.join(PROJECT_ROOT, "InCTRL")
MODEL_CONFIG = os.path.join(REPO_ROOT, "open_clip", "model_configs", "ViT-B-16-plus-240.json")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "checkpoint.pyth")

IMAGE_SIZE = 240
K_SHOT = 4
OUT_LAYERS = [7, 9, 11]

THRESHOLD_MARGIN = 1.0