import json
import os
import sys
import torch

import config

sys.path.insert(0, os.path.abspath(config.REPO_ROOT))
import open_clip
from open_clip.model import InCTRL, get_cast_dtype


def _force_cpu_shim():
    def _is_cuda(x):
        return isinstance(x, str) and x.startswith("cuda")

    torch.Tensor.cuda = lambda self, *a, **k: self
    _orig_to = torch.Tensor.to

    def _to_cpu(self, *args, **kwargs):
        args = tuple("cpu" if _is_cuda(a) else a for a in args)
        if _is_cuda(kwargs.get("device")):
            kwargs["device"] = "cpu"
        return _orig_to(self, *args, **kwargs)

    torch.Tensor.to = _to_cpu


def build_model(device):
    if device == "cpu":
        _force_cpu_shim()

    with open(config.MODEL_CONFIG, "r") as f:
        model_cfg = json.load(f)

    class Args:
        shot = config.K_SHOT

    model = InCTRL(
        Args(),
        model_cfg["embed_dim"],
        model_cfg["vision_cfg"],
        model_cfg["text_cfg"],
        quick_gelu=False,
        cast_dtype=get_cast_dtype("fp32"),
    )

    if os.path.isfile(config.CHECKPOINT_PATH):
        state = torch.load(config.CHECKPOINT_PATH, map_location="cpu")
        model.load_state_dict(state, strict=True)
    else:
        print(f"[warn] no checkpoint at {config.CHECKPOINT_PATH}, running with random weights")

    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-16-plus-240")
    return model, tokenizer


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m, tok = build_model(device)
    print(f"model built OK on {device}, params: {sum(p.numel() for p in m.parameters()) / 1e6:.1f}M")