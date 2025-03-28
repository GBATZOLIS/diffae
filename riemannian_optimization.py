#!/usr/bin/env python
"""
riemannian_optimization.py

Performs riemannian optimization in the latent space of an autoencoder/diffusion decoder
(loaded via ffhq128_autoenc_latent) to increase the classifier output for a target attribute
(loaded via ffhq128_autoenc_cls).

We:
  1. Load the autoencoder with latent diffusion (LitModel) from templates_latent.py.
  2. Load the classifier model from templates_cls.py.
  3. Pick an image from the dataset, encode it to latent space, and also get a stochastic sample xT.
  4. Define a classifier-based objective function in the latent space.
  5. Build a DiffusionWrapper for the latent diffusion geometry and run the riemannian optimizer.
  6. Decode the manipulated latent with xT to produce the final image.
"""

import os
import time
import math
import numpy as np
import torch
import torch.multiprocessing as mp
from argparse import ArgumentParser
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ---------------------------
# Import your code's modules
# ---------------------------
# The user indicates there's no config folder, but we do have templates, templates_latent, templates_cls.
from templates_latent import ffhq128_autoenc_latent   # the autoencoder config w/ latent diffusion
from templates_cls import ffhq128_autoenc_cls         # the classifier config
# The main experiment & classifier code
from experiment import LitModel
from experiment_classifier import ClsModel
# The dataset code
from dataset import ImageDataset, CelebAttrDataset

# Riemannian optimization & geometry library
from data_geometry.optim_function import get_optim_function
from data_geometry.riemannian_optimization.retraction import create_retraction_fn
from data_geometry.riemannian_optimization.optimizers import get_riemannian_optimizer
from data_geometry.utils.visualization import visualize_riemannian_optimization_selector

# ---------------------------
# Helper Functions
# ---------------------------
def flatten_tensor(x):
    """Flattens a tensor across all dimensions except the batch dimension."""
    return x.view(x.size(0), -1)

def unflatten_tensor(x, orig_shape):
    """Reshapes a flattened tensor back to (B, *orig_shape)."""
    return x.view(x.size(0), *orig_shape)

def ensure_time_tensor(t, batch_size):
    """
    Ensures t is a 1D tensor of shape (batch_size,).
    If t is a scalar, expands it.
    """
    if t.dim() == 0:
        return t.expand(batch_size)
    elif t.dim() == 1:
        return t if t.size(0) == batch_size else t.expand(batch_size)
    else:
        if t.size(-1) == 1:
            return t.squeeze(-1)
        return t

# ---------------------------
# DiffusionWrapper for latent diffusion
# ---------------------------
class DiffusionWrapper:
    """
    Wraps a diffusion process (the latent diffusion sampler) so that it provides:
      - get_alpha_fn() -> α(t)=sqrt(alphas_cumprod[t])
      - get_sigma_fn() -> σ(t)=sqrt(1 - alphas_cumprod[t])
      - snr(t)= α(t)^2 / σ(t)^2

    We assume the diffusion object has:
       - sqrt_alphas_cumprod (numpy array [T])
       - alphas_cumprod (numpy array [T])
    """
    def __init__(self, diffusion):
        self.diffusion = diffusion

    def get_alpha_fn(self):
        def alpha_fn(t):
            alpha_vals = torch.tensor(self.diffusion.sqrt_alphas_cumprod,
                                      device=t.device, dtype=t.dtype)
            t_idx = t.long()
            return alpha_vals[t_idx].view(t.size(0), 1)
        return alpha_fn

    def get_sigma_fn(self):
        def sigma_fn(t):
            sigma_vals = torch.tensor(np.sqrt(1.0 - self.diffusion.alphas_cumprod),
                                      device=t.device, dtype=t.dtype)
            t_idx = t.long()
            return sigma_vals[t_idx].view(t.size(0), 1)
        return sigma_fn

    def snr(self, t):
        alpha_fn = self.get_alpha_fn()
        sigma_fn = self.get_sigma_fn()
        alpha_t = alpha_fn(t)
        sigma_t = sigma_fn(t)
        return (alpha_t ** 2) / (sigma_t ** 2)

# ---------------------------
# Score and Denoiser for latent space
# ---------------------------
def get_score_fn(diffusion_wrapper, latent_net, t, latent_shape):
    """
    score(z_t) = - noise_pred(z_t,t) / σ(t)
    """
    sigma_fn = diffusion_wrapper.get_sigma_fn()
    def score_fn(z_flat):
        batch_size = z_flat.size(0)
        t_corr = ensure_time_tensor(t, batch_size)
        sigma_t = sigma_fn(t_corr).view(batch_size, 1)
        z = unflatten_tensor(z_flat, latent_shape)
        noise_pred = latent_net(z, t_corr).pred
        noise_pred_flat = flatten_tensor(noise_pred)
        return - noise_pred_flat / sigma_t
    return score_fn

def get_denoiser_fn(diffusion_wrapper, latent_net, t, latent_shape):
    """
    denoiser(z_t) = (z_t - σ(t)*noise_pred(z_t,t)) / α(t)
    """
    alpha_fn = diffusion_wrapper.get_alpha_fn()
    sigma_fn = diffusion_wrapper.get_sigma_fn()
    def denoiser_fn(z_flat):
        batch_size = z_flat.size(0)
        t_corr = ensure_time_tensor(t, batch_size)
        sigma_t = sigma_fn(t_corr).view(batch_size, 1)
        alpha_t = alpha_fn(t_corr).view(batch_size, 1)
        z = unflatten_tensor(z_flat, latent_shape)
        noise_pred = latent_net(z, t_corr).pred
        noise_pred_flat = flatten_tensor(noise_pred)
        return (z_flat - sigma_t * noise_pred_flat) / alpha_t
    return denoiser_fn

# ---------------------------
# Classifier Objective
# ---------------------------
def classifier_objective(x_flat, cls_model, cls_id, latent_shape):
    """
    x_flat is the latent in the autoencoder's "native" (denormalized) space.
    The classifier might have been trained on normalized latents if manipulate_znormalize=True.
    We check that flag and apply normalization if needed.
    Then we compute the logit for the chosen class (cls_id) and return its negative
    so that minimizing the objective = maximizing the logit.
    """
    x = unflatten_tensor(x_flat, latent_shape)
    if getattr(cls_model.conf, "manipulate_znormalize", False):
        x_in = cls_model.normalize(x)
    else:
        x_in = x
    logits = cls_model.classifier(x_in)
    return -logits[:, cls_id]

def compute_discrete_time_from_target_snr(riem_config, autoenc_conf):
    """
    Computes the discrete diffusion time index from a target SNR value provided in the riemannian config.
    
    Steps:
      1. If "target_snr" exists in the config, compute the desired α_cumprod as:
            desired_alpha = target_snr / (1 + target_snr)
      2. Build the latent diffusion process to access its alphas_cumprod schedule.
      3. Find the index (discrete time) whose α_cumprod is closest to desired_alpha.
      4. Print diagnostic information and return the best matching index.
      5. If no "target_snr" is provided, return the default "time_for_perturbation" from the config.
    
    Args:
      riem_config (dict): The riemannian optimization configuration dictionary.
      autoenc_conf: The autoencoder configuration (providing access to latent diffusion parameters).
      
    Returns:
      int or float: The discrete time index for the diffusion process if ro_snr is provided;
                    otherwise, the default time value from the config.
    """
    ro_snr = riem_config["ro_SNR"]
    print(f"SNR at which Riemannian optimization takes place: {ro_snr}")
    # Compute desired α_cumprod = ro_snr / (1 + ro_snr)
    desired_alpha = ro_snr / (1 + ro_snr)
    print(f"Desired alpha_cumprod (ro_snr / (1+ro_snr)): {desired_alpha:.4f}")
        
    # Build the latent diffusion process to access its discrete schedule.
    latent_diffusion = autoenc_conf.make_latent_eval_diffusion_conf().make_sampler()
    alphas_cumprod = latent_diffusion.alphas_cumprod  # assumed to be a numpy array
    
    # Print SNR for the first 10 diffusion time steps
    print("\n--- SNR for first 10 diffusion time steps ---")
    for t in range(15):
        alpha = alphas_cumprod[t]
        sigma_squared = 1.0 - alpha
        snr = alpha / sigma_squared
        print(f"Step {t:2d}: alpha_cumprod = {alpha:.6f}, SNR = {snr:.6f}")
    print("---------------------------------------------\n")

    # Find the index t for which alphas_cumprod is closest to desired_alpha.
    diffs = np.abs(alphas_cumprod - desired_alpha)
    best_t = int(np.argmin(diffs))
    computed_snr = alphas_cumprod[best_t] / (1 - alphas_cumprod[best_t])
    print(f"Closest discrete diffusion time index found: {best_t}")
    print(f"alpha_cumprod at index {best_t}: {alphas_cumprod[best_t]:.4f}")
    print(f"Computed SNR at this index: {computed_snr:.4f}")
    return best_t


# ---------------------------
# Riemannian Config Loader
# ---------------------------
def load_riemannian_config(path):
    """
    Loads a Python file that defines a CONFIG dict.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Riemannian config file not found: {path}")
    config_dict = {}
    with open(path, "r") as f:
        code = compile(f.read(), path, "exec")
        exec(code, config_dict)
    if "CONFIG" not in config_dict:
        raise ValueError(f"No 'CONFIG' dictionary found in {path}")
    return config_dict["CONFIG"]

# ---------------------------
# Main Riemannian Optimization Routine
# ---------------------------
def riemannian_optimization(riem_config_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load the riemannian config ---
    riem_config = load_riemannian_config(riem_config_path)

    # --- Load the autoencoder config w/ latent diffusion ---
    print("Loading autoencoder model (latent config) from templates_latent.py ...")
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
    #model.latent_net.to(device)

    # --- Load the classifier config & model from templates_cls.py ---
    print("Loading classifier model from templates_cls.py ...")
    cls_conf = ffhq128_autoenc_cls()
    cls_model = ClsModel(cls_conf)
    cls_ckpt = os.path.join("checkpoints", cls_conf.name, "last.ckpt")
    print(f"Loading classifier checkpoint from {cls_ckpt}")
    cls_state = torch.load(cls_ckpt, map_location="cpu")
    cls_model.load_state_dict(cls_state["state_dict"], strict=False)
    cls_model.to(device)
    cls_model.eval()

    # --- Load data sample and encode to latent space ---
    print("Loading data sample from 'imgs_align' ...")
    data = ImageDataset("imgs_align", image_size=autoenc_conf.img_size,
                        exts=["jpg", "JPG", "png"], do_augment=False)
    batch = data[0]["img"][None]  # shape (1, C, H, W)
    batch = batch.to(device)

    # Encode to get the "denormalized" latent
    cond = model.encode(batch)  # shape (1, style_ch)
    # Also get a stochastic latent sample xT
    xT = model.encode_stochastic(batch, cond, T=250)
    latent_shape = cond.shape[1:]
    x0_flat = flatten_tensor(cond) #this is the latent that we will optimize
    riem_config["initial_point"] = x0_flat

    t_val = compute_discrete_time_from_target_snr(riem_config, autoenc_conf)
    batch_size = x0_flat.size(0)
    t_latent = torch.full((batch_size,), t_val, dtype=torch.float32, device=device)

    # --- Build the latent diffusion process & wrapper ---
    print("Building latent diffusion process ...")
    latent_diffusion = autoenc_conf.make_latent_eval_diffusion_conf().make_sampler()
    latent_wrapper = DiffusionWrapper(latent_diffusion)
    try:
        snr_val = latent_wrapper.snr(t_latent)
        print(f"Latent diffusion SNR at t={t_val}: {snr_val.mean().item():.4f}")
    except Exception as e:
        print("Could not compute latent SNR:", e)

    # --- Build score and denoiser functions ---
    score_fn = get_score_fn(latent_wrapper, model.ema_model.latent_net, t_latent, latent_shape)
    denoiser_fn = get_denoiser_fn(latent_wrapper, model.ema_model.latent_net, t_latent, latent_shape)
    retraction_fn = create_retraction_fn(
        retraction_type=riem_config.get("retraction_operator", "identity"),
        denoiser_fn=denoiser_fn
    )

    # --- Define the optimization objective ---
    target_class = "Wavy_Hair"
    cls_id = CelebAttrDataset.cls_to_id[target_class]
    print(f"Target class '{target_class}' has id {cls_id}")

    def opt_fn(x_flat):
        return classifier_objective(x_flat, cls_model, cls_id, latent_shape)

    # --- Run the riemannian optimizer ---
    print("Running riemannian optimization on the latent space...")
    riem_opt = get_riemannian_optimizer(score_fn, opt_fn, riem_config, retraction_fn)
    start_time = time.time()
    trajectory, metrics = riem_opt.run()
    elapsed_time = time.time() - start_time
    print(f"Riemannian optimization completed in {elapsed_time:.2f} seconds.")

    # --- Get optimized latent ---
    x_opt_flat = trajectory[-1]
    x_opt = unflatten_tensor(x_opt_flat, latent_shape)
    print("Optimized latent obtained.")

    # --- Render manipulated image ---
    T_render = riem_config.get("T_render", 100)
    manipulated_img = model.render(xT, x_opt, T=T_render)
    manipulated_img = (manipulated_img + 1) / 2.0

    # --- Save results ---
    output_dir = "riem_results"
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

    # --- Visualize the optimization trajectory (optional) ---
    '''
    visualize_riemannian_optimization_selector(
        perturbed_points=x0_flat,
        score_fn=score_fn,
        time_val=t_val,
        trajectory=[t.detach().cpu() for t in trajectory],
        metrics=metrics,
        min_point=np.array(riem_config["min_point"]),
        orig_shape=latent_shape,
        log_dir=output_dir,
        plot_filename=riem_config["plot_filename"]
    )
    '''

# ---------------------------
# Main entry point
# ---------------------------
def main():
    mp.set_start_method("spawn", force=True)
    parser = ArgumentParser(description="Riemannian Optimization on Autoencoder Latent Space")
    parser.add_argument("--ro-config", type=str, required=True,
                        help="Path to the riemannian config file (Python file with CONFIG dict)")
    args = parser.parse_args()

    riemannian_optimization(args.ro_config)

if __name__ == "__main__":
    main()
