# Few-Shot Industrial Anomally Detection

A terminal tool that checks product photos for defects using only 4 defect-free
reference images. Built on **InCTRL** (Zhu & Pang, CVPR 2024): one pretrained model,
no retraining per product. Images are padded to a square so nothing is cropped, and a
leave-one-out rule on the references gives the OK / DEFECT verdict.

The official InCTRL code (https://github.com/mala-lab/InCTRL, Apache License 2.0) is
included unmodified in `InCTRL/`.

Paper: [Toward Generalist Anomaly Detection via In-context Residual Learning with
Few-shot Sample Prompts](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhu_Toward_Generalist_Anomaly_Detection_via_In-context_Residual_Learning_with_Few-shot_CVPR_2024_paper.pdf)

## Requirements

- Python 3.10+
- `pip install torch torchvision numpy scikit-learn pillow ftfy regex timm huggingface_hub fvcore iopath psutil simplejson tqdm rich`

## Pretrained checkpoint

Download `trained_on_visa.zip` from the official InCTRL release:
https://drive.google.com/drive/folders/1mqDC-GSpJEueERPkqP7u8ORLQUU0KdYt

Unzip it and copy `4/checkpoint.pyth` to:

```
checkpoints/
  checkpoint.pyth
```

## Speed up with an NVIDIA GPU (optional)

1. Check your driver with `nvidia-smi`.
2. Install the CUDA build of PyTorch:
   ```
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```
3. Check it worked (should print `True`):
   ```
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## Dataset placement

Make a folder with two subfolders(Already exists in the repo with pencil sample):

```
product/
  reference/   # exactly 4 defect-free images
  query/       # images to inspect
```
Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`.

You can also use the images from `Sample Dataset` folder in the repo.

## Running

```
cd product
python ../proto/run_inspection.py --name pencil
```

`--name` is the object name (e.g. `mug`, `pencil`, `bottle`). It asks whether to use
CUDA or CPU, or you can pass `--device cpu` / `--device cuda`. Other options are
`--ref` and `--query` to use different folder names.