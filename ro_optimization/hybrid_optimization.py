#!/usr/bin/env python
"""
Main Riemannian Optimization routine.
"""

import os
import time
import torch
# Enable TF32 for float32 matrix multiplies on Ampere+ GPUs
torch.set_float32_matmul_precision('high')

import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# Import helper functions from local modules
from .config_loader import load_riemannian_config
from .utils import flatten_tensor, unflatten_tensor
from .diffusion_utils import (
    DiffusionWrapper,
    get_classifier_fn,
    get_score_fn,
    get_denoiser_fn,
    compute_discrete_time_from_target_snr
)
from .objectives import get_opt_fn

# Import project-specific modules
from templates_latent import ffhq128_autoenc_latent
from templates_cls import ffhq128_autoenc_non_linear_cls
from experiment import LitModel
from experiment_classifier import ClsModel
from dataset import ImageDataset, CelebAttrDataset

# Riemannian optimization & geometry library
from data_geometry.optim_function import get_optim_function
from data_geometry.riemannian_optimization.retraction import create_retraction_fn
from data_geometry.riemannian_optimization import get_riemannian_optimizer

# Visualization (original single-phase)
from .visualization_utils import visualize_trajectory, save_gif_from_rendered_images


def riemannian_optimization(riem_config_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the riemannian config
    riem_config = load_riemannian_config(riem_config_path)

    # Load autoencoder model (latent config)
    print("Loading autoencoder model (latent config) ...")
    autoenc_conf = ffhq128_autoenc_latent()
    autoenc_conf.T_eval = 1000
    autoenc_conf.latent_T_eval = 1000
    model = LitModel(autoenc_conf)
    ckpt_path = os.path.join("checkpoints", autoenc_conf.name, "last.ckpt")
    print(f"Loading autoencoder checkpoint from {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["state_dict"], strict=False)
    model.ema_model.eval()
    model.ema_model.to(device)

    if not hasattr(model.ema_model, "latent_net"):
        raise ValueError("Autoencoder model does not contain latent_net.")

    # Load classifier model
    print("Loading classifier model ...")
    cls_conf = ffhq128_autoenc_non_linear_cls()
    cls_model = ClsModel(cls_conf)
    cls_ckpt = os.path.join("checkpoints", cls_conf.name, "last.ckpt")
    print(f"Loading classifier checkpoint from {cls_ckpt}")
    cls_state = torch.load(cls_ckpt, map_location="cpu")
    cls_model.load_state_dict(cls_state["state_dict"], strict=False)
    cls_model.to(device)
    cls_model.eval()

    # Load data sample and encode to latent space
    print("Loading data sample ...")
    data = ImageDataset(
        "imgs_align",
        image_size=autoenc_conf.img_size,
        exts=["jpg", "JPG", "png"],
        do_augment=False
    )
    batch = data[0]["img"].unsqueeze(0).to(device)

    cond = model.encode(batch)
    xT = model.encode_stochastic(batch, cond, T=250)
    latent_shape = cond.shape[1:]
    x0_flat = flatten_tensor(cond)

    # Check normalization flags
    assert getattr(cls_model.conf, "manipulate_znormalize", False)
    assert getattr(model.conf, "latent_znormalize", False)

    # Normalize initial latent
    x0_flat_norm = cls_model.normalize(x0_flat)
    riem_config["initial_point"] = x0_flat_norm

    # Compute target SNR time
    t_val = compute_discrete_time_from_target_snr(riem_config, autoenc_conf)
    t_latent = torch.full((1,), t_val, dtype=torch.float32, device=device)

    # Build diffusion wrapper
    latent_diffusion = autoenc_conf.make_latent_eval_diffusion_conf().make_sampler()
    latent_wrapper = DiffusionWrapper(latent_diffusion)
    try:
        snr_val = latent_wrapper.snr(t_latent)
        print(f"Latent diffusion SNR at t={t_val}: {snr_val.mean().item():.4f}")
    except Exception:
        print("Could not compute latent SNR.")

    # Score, denoiser and retraction
    score_fn = get_score_fn(latent_wrapper, model.ema_model.latent_net, t_latent, latent_shape)
    denoiser_fn = get_denoiser_fn(latent_wrapper, model.ema_model.latent_net, t_latent, latent_shape)
    retraction_fn = create_retraction_fn(
        retraction_type=riem_config.get("retraction_operator", "identity"),
        denoiser_fn=denoiser_fn
    )

    # Define objective
    target_class = "Smiling"
    cls_id = CelebAttrDataset.cls_to_id[target_class]
    classifier_weight = riem_config.get("classifier_weight", 1.0)
    reg_norm_weight    = riem_config.get("reg_norm_weight", 0.5)
    reg_norm_type      = riem_config.get("reg_norm_type", "L2")
    classifier_fn = get_classifier_fn(cls_model, torch.zeros_like(t_latent), latent_shape)
    opt_fn = get_opt_fn(
        classifier_fn, cls_id, latent_shape,
        x0_flat_norm,
        classifier_weight, reg_norm_weight, reg_norm_type
    )

    # ─── two-phase optimization: optional RGD warm-up → final optimizer ───
    print("Running riemannian optimization …")
    warmup_steps = riem_config.get("gd_warmup_steps", 0)
    if riem_config.get("optimizer_type") in ["RiemannianHMC", "RiemannianULA", "RiemannianMALA"] and warmup_steps > 0:
        print(f" → Warm‑up: running RiemannianGD for {warmup_steps} steps")
        gd_config = riem_config.copy()
        gd_config["optimizer_type"]   = "gradient_descent"
        gd_config["riemannian_steps"] = warmup_steps

        gd_opt = get_riemannian_optimizer(score_fn, opt_fn, gd_config, retraction_fn)
        warm_traj, _ = gd_opt.run()
        # Denormalize warm-up latents
        warm_denorm = [cls_model.denormalize(z) for z in warm_traj]

        # feed GD final latent into main run
        riem_config["initial_point"] = warm_traj[-1]
        print(" → Warm‑up done, switching to RHMC.")
    else:
        # no warm-up
        warm_denorm = []

    # final optimizer (HMC or whichever)
    riem_opt = get_riemannian_optimizer(score_fn, opt_fn, riem_config, retraction_fn)
    start_time = time.time()
    trajectory, metrics = riem_opt.run()
    elapsed_time = time.time() - start_time
    print(f"Riemannian optimization completed in {elapsed_time:.2f} seconds.")

    # Denormalize main trajectory
    denorm_traj = [cls_model.denormalize(latent) for latent in trajectory]

    # Save original vs. manipulated comparison
    output_dir = riem_config.get('log_dir', 'logs')
    os.makedirs(output_dir, exist_ok=True)
    T_render = riem_config.get("T_render", 100)

    original_img = model.render(xT, cond, T=T_render)
    original_img = (original_img + 1) / 2.0
    x_opt_denorm = unflatten_tensor(denorm_traj[-1], latent_shape)
    manipulated_img = model.render(xT, x_opt_denorm, T=T_render)
    manipulated_img = (manipulated_img + 1) / 2.0

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(original_img[0].permute(1, 2, 0).cpu().numpy())
    ax[0].set_title("Original")
    ax[0].axis("off")
    ax[1].imshow(manipulated_img[0].permute(1, 2, 0).cpu().numpy())
    ax[1].set_title(f"Manipulated ({target_class})")
    ax[1].axis("off")
    comp_path = os.path.join(output_dir, "comparison.png")
    plt.savefig(comp_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison image saved to {comp_path}")

    # ─── full trajectory viz ─────────────────────────────────
    full_traj = warm_denorm + denorm_traj
    traj_save = os.path.join(output_dir, "trajectory.png")
    rendered = visualize_trajectory(
        model, xT,
        full_traj,
        latent_shape, T_render,
        save_path=traj_save,
        fast_mode=True
    )

    # gif
    gif_path = os.path.join(output_dir, "trajectory.gif")
    save_gif_from_rendered_images(rendered, gif_path, duration_sec=6)


def main():
    mp.set_start_method("spawn", force=True)
    parser = ArgumentParser(description="Riemannian Optimization on Autoencoder Latent Space")
    parser.add_argument(
        "--ro-config", type=str, required=True,
        help="Path to the riemannian config file (Python file with CONFIG dict)"
    )
    args = parser.parse_args()
    riemannian_optimization(args.ro_config)


if __name__ == "__main__":
    main()
