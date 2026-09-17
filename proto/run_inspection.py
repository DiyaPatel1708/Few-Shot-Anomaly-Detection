import argparse
import os
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
from rich.console import Console
from rich.table import Table

import config
import detector as det
from model import build_model
from threshold import compute_threshold


def pick_device(console):
    if not torch.cuda.is_available():
        console.print("[yellow]no CUDA GPU detected -- running on CPU[/yellow]")
        return "cpu"
    choice = input("Run on Cuda(fast) or CPU? Choose cuda only if you have Nvidia's Discrete Graphics\n\t[cuda/CPU, default - CPU]: ").strip().lower()
    return "cuda" if choice.startswith("cuda") else "cpu"


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="reference")
    ap.add_argument("--query", default="query")
    ap.add_argument("--name", required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = ap.parse_args()

    console = Console()
    device = args.device or pick_device(console)
    if device == "cuda" and not torch.cuda.is_available():
        console.print("[red]--device cuda requested but no GPU found, falling back to cpu[/red]")
        device = "cpu"

    start_time = time.perf_counter()
    model, tokenizer = build_model(device)
    detector = det.Detector(model, tokenizer, device)

    ref_paths = det.list_images(args.ref)
    detector.register(ref_paths, args.name)

    threshold, loo_scores, worst_ref = compute_threshold(detector)

    query_paths = det.list_images(args.query)
    if not query_paths:
        console.print(f"[red]no images found in {args.query}[/red]")
        return

    table = Table(title=f"InCTRL inspection: {args.name}  (K={config.K_SHOT}, device={device})")
    table.add_column("image")
    table.add_column("score", justify="right")
    table.add_column("verdict")

    n_defect = 0
    for p in query_paths:
        score = detector.score(p)
        verdict = "DEFECT" if score > threshold else "OK"
        n_defect += verdict == "DEFECT"
        style = "red" if verdict == "DEFECT" else "green"
        table.add_row(os.path.basename(p), f"{score:.3f}", f"[{style}]{verdict}[/{style}]")

    console.print(f"leave-one-out ref scores: {[f'{n}={s:.3f}' for n, s in loo_scores]}  ->  threshold={threshold:.3f}")
    console.print(f"[yellow]most atypical reference photo: {worst_ref}[/yellow]")
    console.print(table)
    console.print(f"{len(query_paths)} inspected | {n_defect} defective")
    console.print(f"total runtime: {time.perf_counter() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()