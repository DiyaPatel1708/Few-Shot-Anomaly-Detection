import glob
import os
import torch
from PIL import Image
from torchvision import transforms

import config

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")

_transform = transforms.Compose([
    transforms.Resize(size=config.IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(size=(config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.Lambda(lambda im: im.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
])


def list_images(folder):
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(paths)


def load_image(path, device):
    return _transform(Image.open(path)).to(device)


class Detector:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def register(self, ref_paths, obj_name):
        assert len(ref_paths) == config.K_SHOT, f"need exactly {config.K_SHOT} reference images, got {len(ref_paths)}"
        self.ref_paths = ref_paths
        self.normal_list = [load_image(p, self.device) for p in ref_paths]
        self.obj_name = obj_name

    @torch.no_grad()
    def score(self, query_path):
        img = load_image(query_path, self.device).unsqueeze(0)
        text = [self.obj_name]
        final_score, _ = self.model(self.tokenizer, [img], text, self.normal_list)
        return final_score.item()

    @torch.no_grad()
    def score_leave_one_out(self):
        scores = []
        full_refs = self.normal_list
        for i in range(len(full_refs)):
            held_out = full_refs[i]
            rest = full_refs[:i] + full_refs[i + 1:]
            img = held_out.unsqueeze(0)
            final_score, _ = self.model(self.tokenizer, [img], [self.obj_name], rest)
            scores.append((os.path.basename(self.ref_paths[i]), final_score.item()))
        return scores