# Few-Shot Industrial Anomaly Detection

A terminal tool that detects and localizes defects in product photos using
only a handful of defect-free reference images. Built on **InCTRL**
(Zhu & Pang, CVPR 2024) — one pretrained model, no retraining per product.

Uses the official InCTRL implementation (https://github.com/mala-lab/InCTRL,
Apache License 2.0), vendored in `InCTRL/`, unmodified.

Paper: [Toward Generalist Anomaly Detection via In-context Residual Learning
with Few-shot Sample Prompts](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhu_Toward_Generalist_Anomaly_Detection_via_In-context_Residual_Learning_with_Few-shot_CVPR_2024_paper.pdf)

## Requirements

- Python 3.10+
- `pip install torch torchvision numpy scikit-learn pillow ftfy regex timm huggingface_hub fvcore iopath psutil simplejson tqdm rich`

## Pretrained checkpoint (required)

Download `trained_on_visa.zip` from the official InCTRL release:
https://drive.google.com/drive/folders/1McmfxF8_H0BeRvcJ_poGIB-ATQCDDEIa

Unzip it, take the `4/checkpoint.pyth` file, and place it at:

```
Your Project Directory/
              checkpoints/
                checkpoint.pyth
```

## Speed up with an NVIDIA GPU (optional)

If you have an NVIDIA GPU, install the CUDA build of PyTorch instead of the
default CPU-only one:

1. Make sure you have a recent NVIDIA driver installed (check with `nvidia-smi`).
2. Install the CUDA-enabled PyTorch build:
   ```
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```
3. Verify it's working:
   ```
   python -c "import torch; print(torch.cuda.is_available())"
   ```
   Should print `True`.

No code changes needed — the tool asks at runtime whether to use CUDA or CPU.

## Dataset placement

Create a folder with two subfolders:

```
product/
  reference/   # exactly 4 defect-free images of the product
  query/       # images you want to inspect
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`.

## Running

```
cd product
python /path/to/proto/run_inspection.py --name <object_name>
```

`--name` is a plain word describing the object (e.g. `mug`, `pencil`, `bottle`).

Optional flags:
- `--ref <folder>` / `--query <folder>` — override the default `reference`/`query` folder names
- `--device cpu` / `--device cuda` — skip the interactive prompt and force a device

Example, using the included sample data:
```
cd sample
python ../proto/run_inspection.py --name pencil
```
