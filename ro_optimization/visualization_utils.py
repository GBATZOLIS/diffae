import matplotlib.pyplot as plt
import numpy as np
import torch
from .utils import unflatten_tensor

def visualize_trajectory(model, xT, trajectory, latent_shape, T_render, save_path, fast_mode=False):
    """
    Visualizes the optimization trajectory by rendering images at each optimization step.
    """
    n_steps = len(trajectory)
    batch_size = trajectory[0].size(0)
    rendered_images = []

    if not fast_mode:
        for step in range(n_steps):
            latent_flat = trajectory[step]
            latent_unflat = unflatten_tensor(latent_flat, latent_shape)
            imgs = model.render(xT, latent_unflat, T=T_render)
            imgs = (imgs + 1) / 2.0
            imgs_np = imgs.cpu().permute(0, 2, 3, 1).numpy()
            rendered_images.append(imgs_np)
    else:
        all_latents = torch.cat([unflatten_tensor(latent, latent_shape) for latent in trajectory], dim=0)
        xT_repeated = xT.repeat(n_steps, 1, 1, 1)
        imgs = model.render(xT_repeated, all_latents, T=T_render)
        imgs = (imgs + 1) / 2.0
        imgs = imgs.view(n_steps, batch_size, *imgs.shape[1:])
        imgs = imgs.permute(0, 1, 3, 4, 2).cpu().numpy()
        for step in range(n_steps):
            rendered_images.append(imgs[step])
    
    fig, axes = plt.subplots(batch_size, n_steps, figsize=(n_steps * 3, batch_size * 3))
    if batch_size == 1 and n_steps == 1:
        axes = np.array([[axes]])
    elif batch_size == 1:
        axes = np.expand_dims(axes, axis=0)
    elif n_steps == 1:
        axes = np.expand_dims(axes, axis=1)
    
    for i in range(batch_size):
        for j in range(n_steps):
            ax = axes[i, j]
            ax.imshow(rendered_images[j][i])
            ax.axis('off')
            if i == 0:
                ax.set_title(f"Step {j}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
