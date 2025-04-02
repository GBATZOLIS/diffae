#!/usr/bin/env python
"""
Main Riemannian Optimization routine.
"""

import os
import time
import torch
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# Import helper functions from local modules
from .config_loader import load_riemannian_config
from .utils import flatten_tensor, unflatten_tensor
from .diffusion_utils import DiffusionWrapper, get_score_fn, get_denoiser_fn, compute_discrete_time_from_target_snr
from .objectives import get_opt_fn
from .visualization_utils import visualize_trajectory

# Import project-specific modules (unchanged)
from templates_latent import ffhq128_autoenc_latent   # autoencoder config
from templates_cls import ffhq128_autoenc_cls         # classifier config
from experiment import LitModel
from experiment_classifier import ClsModel
from dataset import ImageDataset, CelebAttrDataset

# Riemannian optimization & geometry library (external)
from data_geometry.optim_function import get_optim_function
from data_geometry.riemannian_optimization.retraction import create_retraction_fn
from data_geometry.riemannian_optimization.optimizers import get_riemannian_optimizer

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
    cls_conf = ffhq128_autoenc_cls()
    cls_model = ClsModel(cls_conf)
    cls_ckpt = os.path.join("checkpoints", cls_conf.name, "last.ckpt")
    print(f"Loading classifier checkpoint from {cls_ckpt}")
    cls_state = torch.load(cls_ckpt, map_location="cpu")
    cls_model.load_state_dict(cls_state["state_dict"], strict=False)
    cls_model.to(device)
    cls_model.eval()

    # Load data sample and encode to latent space
    print("Loading data sample ...")
    data = ImageDataset("imgs_align", image_size=autoenc_conf.img_size,
                        exts=["jpg", "JPG", "png"], do_augment=False)
    batch = data[0]["img"][None]
    batch = batch.to(device)

    cond = model.encode(batch)
    xT = model.encode_stochastic(batch, cond, T=250)
    latent_shape = cond.shape[1:]
    x0_flat = flatten_tensor(cond)

    # Check that both models expect normalized latents.
    assert getattr(cls_model.conf, "manipulate_znormalize", False) == True and getattr(model.conf, "latent_znormalize", False) == True, \
        f"cls_model.conf.manipulate_znormalize:{cls_model.conf.manipulate_znormalize} and model.conf.latent_znormalize:{model.conf.latent_znormalize}. Both must be True."

    # Normalize the latent before optimization.
    x0_flat_normalized = cls_model.normalize(x0_flat)
    riem_config["initial_point"] = x0_flat_normalized

    t_val = compute_discrete_time_from_target_snr(riem_config, autoenc_conf)
    batch_size = x0_flat_normalized.size(0)
    t_latent = torch.full((batch_size,), t_val, dtype=torch.float32, device=device)

    # Build latent diffusion process & wrapper
    print("Building latent diffusion process ...")
    latent_diffusion = autoenc_conf.make_latent_eval_diffusion_conf().make_sampler()
    latent_wrapper = DiffusionWrapper(latent_diffusion)
    try:
        snr_val = latent_wrapper.snr(t_latent)
        print(f"Latent diffusion SNR at t={t_val}: {snr_val.mean().item():.4f}")
    except Exception as e:
        print("Could not compute latent SNR:", e)

    # Build score and denoiser functions
    score_fn = get_score_fn(latent_wrapper, model.ema_model.latent_net, t_latent, latent_shape)
    denoiser_fn = get_denoiser_fn(latent_wrapper, model.ema_model.latent_net, t_latent, latent_shape)
    retraction_fn = create_retraction_fn(
        retraction_type=riem_config.get("retraction_operator", "identity"),
        denoiser_fn=denoiser_fn
    )

    # Define optimization objective
    target_class = "Smiling"
    cls_id = CelebAttrDataset.cls_to_id[target_class]
    print(f"Target class '{target_class}' has id {cls_id}")
    l2_lambda = riem_config.get("l2_lambda", 0.1)
    opt_fn = get_opt_fn(cls_model, cls_id, latent_shape, x0_flat_normalized, l2_lambda)

    # Run the riemannian optimizer
    print("Running riemannian optimization ...")
    riem_opt = get_riemannian_optimizer(score_fn, opt_fn, riem_config, retraction_fn)
    start_time = time.time()
    trajectory, metrics = riem_opt.run()
    elapsed_time = time.time() - start_time
    print(f"Riemannian optimization completed in {elapsed_time:.2f} seconds.")

    # Denormalize each latent in the trajectory (using a for loop)
    denorm_trajectory = [cls_model.denormalize(latent) for latent in trajectory]

    # Use the last denormalized latent for final rendering.
    print("Optimized latent obtained.")
    x_opt_denormalized = unflatten_tensor(denorm_trajectory[-1], latent_shape)
    T_render = riem_config.get("T_render", 100)
    manipulated_img = model.render(xT, x_opt_denormalized, T=T_render)
    manipulated_img = (manipulated_img + 1) / 2.0

    # Save results
    output_dir = "ro_optimization/ro_results"
    os.makedirs(output_dir, exist_ok=True)
    original_img = model.render(xT, cond, T=T_render)
    original_img = (original_img + 1) / 2.0

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(original_img[0].permute(1, 2, 0).cpu().numpy())
    ax[0].set_title("Original")
    ax[0].axis("off")
    ax[1].imshow(manipulated_img[0].permute(1, 2, 0).cpu().numpy())
    ax[1].set_title(f"Manipulated ({target_class})")
    ax[1].axis("off")
    comp_path = os.path.join(output_dir, "comparison.png")
    plt.savefig(comp_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Comparison image saved to {comp_path}")

    # Visualize the optimization trajectory using the pre-denormalized latents.
    traj_save_path = os.path.join(output_dir, "trajectory.png")
    visualize_trajectory(model, xT, denorm_trajectory, latent_shape, T_render, traj_save_path, fast_mode=True)

def main():
    mp.set_start_method("spawn", force=True)
    parser = ArgumentParser(description="Riemannian Optimization on Autoencoder Latent Space")
    parser.add_argument("--ro-config", type=str, required=True,
                        help="Path to the riemannian config file (Python file with CONFIG dict)")
    args = parser.parse_args()
    riemannian_optimization(args.ro_config)

if __name__ == "__main__":
    main()
