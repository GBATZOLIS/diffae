#!/usr/bin/env python
"""
Evaluate classifier accuracy across diffusion timesteps,
using only the trained time-dependent ClsModel (which wraps the autoencoder & classifier).
Saves two separate plots:
  1) Accuracy vs Diffusion Time
  2) Accuracy vs SNR (ordered high→low, log-scaled)
"""
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

# Project imports
from config import *
from dataset import *
from templates_cls import ffhq128_autoenc_non_linear_time_cls
from experiment_classifier import ClsModel


def load_cls_model(device):
    # Load the time-dependent latent classifier (which includes the autoencoder)
    cls_conf = ffhq128_autoenc_non_linear_time_cls()
    model = ClsModel(cls_conf)
    ckpt_path = os.path.join("checkpoints", cls_conf.name, "last.ckpt")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state['state_dict'], strict=False)
    model.eval().to(device)
    return model, cls_conf


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load classifier model and config
    cls_model, cls_conf = load_cls_model(device)

    # Extract diffusion parameters from the classifier
    max_t = cls_model.max_diffusion_time
    # alpha_vals and sigma_vals are sqrt(alphas_cumprod) and sqrt(1 - alphas_cumprod)
    alpha_tensor = cls_model.alpha_vals[:max_t].to(device)
    sigma_tensor = cls_model.sigma_vals[:max_t].to(device)

    # Prepare dataset and loader (exactly as training)
    dataset = cls_model.load_dataset()
    loader_native = cls_conf.make_loader(
        dataset,
        shuffle=False,
        num_worker=args.num_workers,
        drop_last=False,
        batch_size=args.batch_size
    )
    total_images = min(args.max_images, len(dataset))
    total_batches = math.ceil(total_images / args.batch_size)
    loader = tqdm(loader_native, total=total_batches, desc="Batches")

    # Initialize counters
    acc = np.zeros(max_t, dtype=np.float64)
    count = np.zeros(max_t, dtype=np.int64)
    seen = 0

    for batch in loader:
        imgs = batch['img'].to(device)
        labels = batch['labels'].to(device)
        bs = imgs.size(0)
        if seen + bs > args.max_images:
            bs = args.max_images - seen
            imgs = imgs[:bs]
            labels = labels[:bs]
        seen += bs

        # Encode to latent via the classifier's ema_model
        with torch.no_grad():
            cond = cls_model.ema_model.encoder(imgs)
            if cls_conf.manipulate_znormalize:
                cond = cls_model.normalize(cond)

        # Evaluate for each diffusion time
        for t in range(max_t):
            a = alpha_tensor[t]
            s = sigma_tensor[t]
            noise = torch.randn_like(cond)
            pert = a * cond + s * noise

            flat = pert.view(bs, -1)
            logits = cls_model.classifier(flat, t=torch.full((bs,), t, device=device))
            preds = (torch.sigmoid(logits) > 0.5).float()

            cls_id = CelebAttrDataset.cls_to_id[args.attribute]
            correct = (preds[:, cls_id] == (labels[:, cls_id] > 0)).sum().item()
            acc[t] += correct
            count[t] += bs

        if seen >= args.max_images:
            break

    # Compute accuracy curve
    accuracy = acc / count

    # 1) Plot Accuracy vs Diffusion Time
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(max_t), accuracy)
    plt.title(f"Accuracy vs Diffusion Time for {args.attribute}")
    plt.xlabel("Diffusion Time (t)")
    plt.ylabel("Accuracy")
    out_time = args.output_path.replace('.png', '_vs_time.png')
    os.makedirs(os.path.dirname(out_time), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_time)
    print(f"Saved plot to {out_time}")
    plt.close()

    # 2) Plot Accuracy vs SNR (high→low) on log x-axis
    # Compute SNR = (alpha^2)/(sigma^2)
    snr_vals = (alpha_tensor.cpu().numpy()**2) / (sigma_tensor.cpu().numpy()**2)
    # Sort descending to get high→low
    idx = np.argsort(snr_vals)[::-1]
    snr_sorted = snr_vals[idx]
    acc_sorted = accuracy[idx]

    plt.figure(figsize=(8, 4))
    plt.plot(snr_sorted, acc_sorted)
    plt.xscale('log')
    plt.gca().invert_xaxis()
    plt.title(f"Accuracy vs SNR for {args.attribute}")
    plt.xlabel("SNR (high → low, log scale)")
    plt.ylabel("Accuracy")
    out_snr = args.output_path.replace('.png', '_vs_snr.png')
    plt.tight_layout()
    plt.savefig(out_snr)
    print(f"Saved plot to {out_snr}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate classifier accuracy across diffusion timesteps")
    parser.add_argument('--max-images', type=int, default=5000, help='Number of images to evaluate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--attribute', type=str, default='Male', help='Target attribute name')
    parser.add_argument('--output-path', type=str, default='ro_optimization/ro_results/infer_RO_start/accuracy.png', help='Base path to save the plot PNGs')
    args = parser.parse_args()
    main(args)
