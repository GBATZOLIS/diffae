import matplotlib.pyplot as plt
import numpy as np
import torch
from .utils import unflatten_tensor
import imageio
import math

def visualize_trajectory(model, xT, trajectory, latent_shape, T_render, save_path, fast_mode=False):
    """
    Visualizes the optimization trajectory by rendering images at each optimization step.
    Also returns the rendered images for optional gif creation.
    """
    n_steps = len(trajectory)
    batch_size = trajectory[0].size(0)
    rendered_images = []

    if not fast_mode:
        for step in range(n_steps):
            latent_flat = trajectory[step]
            latent_unflat = unflatten_tensor(latent_flat, latent_shape)
            imgs = model.render(xT, latent_unflat, T=T_render)
            imgs = (imgs + 1) / 2.0  # normalize to [0, 1]
            imgs_np = imgs.cpu().permute(0, 2, 3, 1).numpy()  # (B, H, W, 3)
            rendered_images.append(imgs_np)
    else:
        all_latents = torch.cat([unflatten_tensor(latent, latent_shape) for latent in trajectory], dim=0)
        xT_repeated = xT.repeat(n_steps, 1, 1, 1)
        imgs = model.render(xT_repeated, all_latents, T=T_render)
        imgs = (imgs + 1) / 2.0
        imgs = imgs.view(n_steps, batch_size, *imgs.shape[1:])
        imgs_np = imgs.permute(0, 1, 3, 4, 2).cpu().numpy()  # (n_steps, B, H, W, 3)
        rendered_images = list(imgs_np)

    # Plot grid
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

    return rendered_images  # shape: [n_steps][batch_size][H][W][3]



def save_gif_from_rendered_images(rendered_images, gif_path, duration_sec=15):
    """
    Save a GIF showing a grid of samples evolving over optimization steps.
    Arranges each frame into a nearly square (R x C) grid of images.
    """
    num_frames = len(rendered_images)
    batch_size = rendered_images[0].shape[0]
    H, W, C = rendered_images[0][0].shape
    fps = max(1, num_frames // duration_sec)

    # Compute grid size as close to square as possible
    grid_cols = math.ceil(math.sqrt(batch_size))
    grid_rows = math.ceil(batch_size / grid_cols)

    print(f"Saving square grid GIF with {num_frames} frames of {grid_rows}x{grid_cols} layout...")

    frames = []
    for t in range(num_frames):
        imgs = rendered_images[t]  # shape: (B, H, W, 3)
        # Pad with black images if needed
        padded_imgs = list(imgs)
        while len(padded_imgs) < grid_rows * grid_cols:
            padded_imgs.append(np.zeros((H, W, C), dtype=imgs.dtype))

        # Build grid row by row
        grid_rows_list = []
        for r in range(grid_rows):
            row_imgs = padded_imgs[r * grid_cols:(r + 1) * grid_cols]
            row = np.concatenate(row_imgs, axis=1)  # concat horizontally
            grid_rows_list.append(row)
        frame = np.concatenate(grid_rows_list, axis=0)  # concat vertically

        frame = (frame * 255).astype(np.uint8)
        frames.append(frame)

    imageio.mimsave(gif_path, frames, fps=fps)
    print(f"Square grid GIF saved to {gif_path}")
