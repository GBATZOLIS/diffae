# Enable automatic module reloading (Jupyter equivalent of %autoreload)
import importlib

def autoreload():
    importlib.reload(importlib)

autoreload()

# Import necessary modules
import torch
import matplotlib.pyplot as plt
from templates import *
from templates_latent import *
import os

# Set device
device = 'cuda:0'

# Load model configuration
conf = ffhq128_autoenc_latent()
conf.T_eval = 100
conf.latent_T_eval = 100

# Initialize model
model = LitModel(conf)

# Load checkpoint
checkpoint_path = f'checkpoints/{conf.name}/last.ckpt'
state = torch.load(checkpoint_path, map_location='cpu')
print(model.load_state_dict(state['state_dict'], strict=False))

# Move model to device
model.to(device)

# Set random seed
torch.manual_seed(4)

# Generate images
imgs = model.sample(8, device=device, T=20, T_latent=200)

# Create directory if it doesn't exist
output_dir = "results/samples"
os.makedirs(output_dir, exist_ok=True)

# Save images to a file
output_path = os.path.join(output_dir, "generated_samples.png")
fig, ax = plt.subplots(2, 4, figsize=(4*5, 2*5))
ax = ax.flatten()

for i in range(len(imgs)):
    ax[i].imshow(imgs[i].cpu().permute([1, 2, 0]))
    ax[i].axis('off') 
