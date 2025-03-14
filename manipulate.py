# Enable automatic module reloading (Jupyter equivalent of %autoreload)
import importlib
import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import os

from templates import *
from templates_cls import *
from experiment_classifier import ClsModel
from torchvision.utils import save_image

# Set device
device = 'cuda:0'

# Load main model
conf = ffhq128_autoenc_130M()
model = LitModel(conf)

# Load checkpoint
checkpoint_path = f'checkpoints/{conf.name}/last.ckpt'
state = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device)

# Load classifier model
cls_conf = ffhq128_autoenc_cls()
cls_model = ClsModel(cls_conf)

# Load classifier checkpoint
cls_checkpoint_path = f'checkpoints/{cls_conf.name}/last.ckpt'
state = torch.load(cls_checkpoint_path, map_location='cpu')
print('Latent step:', state['global_step'])
cls_model.load_state_dict(state['state_dict'], strict=False)
cls_model.to(device)

# Load dataset and encode image
data = ImageDataset('imgs_align', image_size=conf.img_size, exts=['jpg', 'JPG', 'png'], do_augment=False)
print(data)
batch = data[1]['img'][None]
cond = model.encode(batch.to(device))
xT = model.encode_stochastic(batch.to(device), cond, T=250)

# Create results directory if not exists
output_dir = "results/imgs"
os.makedirs(output_dir, exist_ok=True)

# Save original and encoded images
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ori = (batch + 1) / 2
ax[0].imshow(ori[0].permute(1, 2, 0).cpu())
ax[1].imshow(xT[0].permute(1, 2, 0).cpu())
plt.savefig(os.path.join(output_dir, 'original_vs_encoded.png'))
plt.close()

# Attribute manipulation
print(CelebAttrDataset.id_to_cls)
cls_id = CelebAttrDataset.cls_to_id['Wavy_Hair']
cond2 = cls_model.normalize(cond)
cond2 = cond2 + 0.3 * math.sqrt(512) * F.normalize(cls_model.classifier.weight[cls_id][None, :], dim=1)
cond2 = cls_model.denormalize(cond2)

# Render manipulated image
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
img = model.render(xT, cond2, T=100)
ax[0].imshow(ori[0].permute(1, 2, 0).cpu())
ax[1].imshow(img[0].permute(1, 2, 0).cpu())
plt.savefig(os.path.join(output_dir, 'compare.png'))
plt.close()

# Save manipulated image
save_image(img[0], os.path.join(output_dir, 'output.png'))

print(f"Images saved in {output_dir}")
