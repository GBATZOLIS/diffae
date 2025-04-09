import torch
import numpy as np
from .utils import unflatten_tensor, ensure_time_tensor

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

def get_score_fn(diffusion_wrapper, latent_net, t, latent_shape):
    """
    Returns a score function:
       score(z_t) = - noise_pred(z_t,t) / σ(t)
    """
    sigma_fn = diffusion_wrapper.get_sigma_fn()
    def score_fn(z_flat):
        batch_size = z_flat.size(0)
        t_corr = ensure_time_tensor(t, batch_size)
        sigma_t = sigma_fn(t_corr).view(batch_size, 1)
        z = unflatten_tensor(z_flat, latent_shape)
        noise_pred = latent_net(z, t_corr).pred
        noise_pred_flat = noise_pred.view(batch_size, -1)
        return - noise_pred_flat / sigma_t
    return score_fn

def get_denoiser_fn(diffusion_wrapper, latent_net, t, latent_shape):
    """
    Returns a denoiser function:
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
        noise_pred_flat = noise_pred.view(batch_size, -1)
        return (z_flat - sigma_t * noise_pred_flat) / alpha_t
    return denoiser_fn

def get_classifier_fn(cls_model, t, latent_shape):
    """
    Constructs and returns a classifier function that, when provided with a
    flattened latent vector x_flat, unflattens it to shape (B, *latent_shape) and
    then computes the logits using the classifier for diffusion time t.
    
    Args:
        cls_model: The classifier model. It must have a forward method that accepts
                   two arguments: (x, t). If not time-conditioned, the model should
                   be modified to accept an extra (optional) t.
        t: Either a scalar or a tensor containing the diffusion time(s) at which the
           classifier should operate.
        latent_shape: The shape that x_flat should be reshaped to before passing to the classifier.
        
    Returns:
        A function classifier_fn(x_flat) that returns the raw logits computed by the classifier.
    """
    def classifier_fn(x_flat):
        # Unflatten the latent into the original shape.
        x = unflatten_tensor(x_flat, latent_shape)
        # Forward pass through the classifier.
        # We assume the classifier's forward method has been modified to accept a 't' parameter.
        return cls_model(x, t)
    return classifier_fn

def compute_discrete_time_from_target_snr(riem_config, autoenc_conf):
    """
    Computes the discrete diffusion time index from a target SNR value provided in the riemannian config.
    If no "target_snr" is provided, returns the default "time_for_perturbation" from the config.
    """
    ro_snr = riem_config["ro_SNR"]
    print(f"SNR at which Riemannian optimization takes place: {ro_snr}")
    desired_alpha = ro_snr / (1 + ro_snr)
    print(f"Desired alpha_cumprod (ro_SNR / (1+ro_SNR)): {desired_alpha:.4f}")
    
    latent_diffusion = autoenc_conf.make_latent_eval_diffusion_conf().make_sampler()
    alphas_cumprod = latent_diffusion.alphas_cumprod  # assumed to be a numpy array
    
    print("\n--- SNR for first 10 diffusion time steps ---")
    for t in range(15):
        alpha = alphas_cumprod[t]
        sigma_squared = 1.0 - alpha
        snr = alpha / sigma_squared
        print(f"Step {t:2d}: alpha_cumprod = {alpha:.6f}, SNR = {snr:.6f}")
    print("---------------------------------------------\n")

    diffs = np.abs(alphas_cumprod - desired_alpha)
    best_t = int(np.argmin(diffs))
    computed_snr = alphas_cumprod[best_t] / (1 - alphas_cumprod[best_t])
    print(f"Closest discrete diffusion time index found: {best_t}")
    print(f"alpha_cumprod at index {best_t}: {alphas_cumprod[best_t]:.4f}")
    print(f"Computed SNR at this index: {computed_snr:.4f}")
    return best_t
