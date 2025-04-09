from config import *
from dataset import *
import pandas as pd
import json
import os
import copy
import math

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import *
import torch
import torch.nn as nn
import torch.nn.functional as F

##############################
# Helper Functions
##############################
def get_timestep_embedding(timesteps, embedding_dim, max_period=10000):
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(0, half_dim, device=timesteps.device, dtype=torch.float32) / half_dim
    emb = timesteps.float().unsqueeze(1) * torch.exp(exponent.unsqueeze(0))
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

def compute_discrete_time_from_target_snr(autoenc_conf, target_snr):
    desired_alpha = target_snr / (1 + target_snr)
    latent_diffusion = autoenc_conf.make_latent_eval_diffusion_conf().make_sampler()
    alphas_cumprod = latent_diffusion.alphas_cumprod  # assumed to be a numpy array
    diffs = np.abs(alphas_cumprod - desired_alpha)
    best_t = int(np.argmin(diffs))
    computed_snr = alphas_cumprod[best_t] / (1 - alphas_cumprod[best_t])
    print(f"Closest discrete diffusion time index found: {best_t}")
    print(f"alpha_cumprod at index {best_t}: {alphas_cumprod[best_t]:.4f}")
    print(f"Computed SNR at this index: {computed_snr:.4f}")
    return best_t

##############################
# Utility Classes
##############################
class ZipLoader:
    def __init__(self, loaders):
        self.loaders = loaders
    def __len__(self):
        return len(self.loaders[0])
    def __iter__(self):
        for each in zip(*self.loaders):
            yield each

class FlexibleClassifier(nn.Module):
    def __init__(self, in_features, num_classes, hidden_dims=[], dropout=0.25):
        super().__init__()
        layers = []
        prev_dim = in_features
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.LayerNorm(dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

####################################
# Diffusion Time–Dependent Classifier Model
####################################
class ClsModel(pl.LightningModule):
    def __init__(self, conf: TrainConfig):
        super().__init__()
        assert conf.train_mode.is_manipulate()
        if conf.seed is not None:
            pl.seed_everything(conf.seed)
        self.save_hyperparameters(conf.as_dict_jsonable())
        self.conf = conf

        ####################################
        # Load Base Autoencoder & Latent Statistics
        ####################################
        if conf.train_mode == TrainMode.manipulate:
            self.model = conf.make_model_conf().make_model()
            self.ema_model = copy.deepcopy(self.model)
            self.model.requires_grad_(False)
            self.ema_model.requires_grad_(False)
            self.ema_model.eval()
            if conf.pretrain is not None:
                print(f'loading pretrain ... {conf.pretrain.name}')
                state = torch.load(conf.pretrain.path, map_location="cpu")
                print('step:', state['global_step'])
                self.load_state_dict(state['state_dict'], strict=False)
            if conf.manipulate_znormalize:
                print('loading latent stats ...')
                state = torch.load(conf.latent_infer_path)
                self.conds = state['conds']
                self.register_buffer('conds_mean', state['conds_mean'][None, :])
                self.register_buffer('conds_std', state['conds_std'][None, :])
            else:
                self.conds_mean = None
                self.conds_std = None

        ####################################
        # Determine Number of Classes
        ####################################
        if conf.manipulate_mode in [ManipulateMode.celebahq_all]:
            num_cls = len(CelebAttrDataset.id_to_cls)
        elif conf.manipulate_mode.is_single_class():
            num_cls = 1
        else:
            raise NotImplementedError()

        ####################################
        # Build the Classifier Network
        ####################################
        self.diffusion_time_dependent = getattr(conf, 'diffusion_time_dependent_classifier', False)
        if self.diffusion_time_dependent:
            # Set time embedding dimension and create a time-embedding MLP.
            self.time_embedding_dim = getattr(conf, 'time_embedding_dim', 128)
            self.time_embed = nn.Sequential(
                nn.Linear(self.time_embedding_dim, self.time_embedding_dim),
                nn.ReLU(),
                nn.Linear(self.time_embedding_dim, self.time_embedding_dim)
            )
            # Increase the input dimension to include the time embedding.
            input_dim = conf.style_ch + self.time_embedding_dim
            # Expect the full autoencoder config to be provided as conf.autoenc_config.
            self.autoenc_config = conf.autoenc_config  
            # Precompute the latent diffusion sampler’s kernel arrays once.
            latent_diffusion = self.autoenc_config.make_latent_eval_diffusion_conf().make_sampler()
            # NOTE: We rely on the diffusion object's arrays:
            self.alpha_vals = torch.tensor(latent_diffusion.sqrt_alphas_cumprod,
                                           device=torch.device("cpu"), dtype=torch.float32)
            self.sigma_vals = torch.tensor(np.sqrt(1.0 - latent_diffusion.alphas_cumprod),
                                           device=torch.device("cpu"), dtype=torch.float32)
            # Determine the maximum diffusion time for training.
            # If conf.lower_trainable_snr exists, use compute_discrete_time_from_target_snr to determine the corresponding discrete time.
            self.max_diffusion_time = (
                compute_discrete_time_from_target_snr(self.autoenc_config, conf.lower_trainable_snr)
                if getattr(conf, 'lower_trainable_snr', None) is not None 
                else 1000
            )
        else:
            input_dim = conf.style_ch

        if conf.train_mode == TrainMode.manipulate:
            classifier_type = getattr(conf, 'classifier_type', 'linear')
            if classifier_type == 'linear':
                self.classifier = nn.Linear(input_dim, num_cls)
            elif classifier_type == 'nonlinear':
                hidden_dims = getattr(conf, 'non_linear_hidden_dims', [])
                dropout = getattr(conf, 'non_linear_dropout', 0.2)
                self.classifier = FlexibleClassifier(input_dim, num_cls,
                                                     hidden_dims=hidden_dims,
                                                     dropout=dropout)
            else:
                raise ValueError(f"Unknown classifier_type: {classifier_type}")
        else:
            raise NotImplementedError()

        self.ema_classifier = copy.deepcopy(self.classifier)

    def state_dict(self, *args, **kwargs):
        out = {}
        for k, v in super().state_dict(*args, **kwargs).items():
            if k.startswith('model.') or k.startswith('ema_model.'):
                continue
            else:
                out[k] = v
        return out

    def load_state_dict(self, state_dict, strict: bool = None):
        if self.conf.train_mode == TrainMode.manipulate:
            if strict is None:
                strict = False
        else:
            if strict is None:
                strict = True
        return super().load_state_dict(state_dict, strict=strict)

    def normalize(self, cond):
        return (cond - self.conds_mean.to(self.device)) / self.conds_std.to(self.device)

    def denormalize(self, cond):
        return cond * self.conds_std.to(self.device) + self.conds_mean.to(self.device)

    ####################################
    # Dataset and DataLoader Methods
    ####################################
    def load_dataset(self):
        if self.conf.manipulate_mode == ManipulateMode.d2c_fewshot:
            return CelebD2CAttrFewshotDataset(
                cls_name=self.conf.manipulate_cls,
                K=self.conf.manipulate_shots,
                img_folder=data_paths['celeba'],
                img_size=self.conf.img_size,
                seed=self.conf.manipulate_seed,
                all_neg=False,
                do_augment=True,
            )
        elif self.conf.manipulate_mode == ManipulateMode.d2c_fewshot_allneg:
            img_folder = data_paths['celeba']
            return [
                CelebD2CAttrFewshotDataset(
                    cls_name=self.conf.manipulate_cls,
                    K=self.conf.manipulate_shots,
                    img_folder=img_folder,
                    img_size=self.conf.img_size,
                    only_cls_name=self.conf.manipulate_cls,
                    only_cls_value=1,
                    seed=self.conf.manipulate_seed,
                    all_neg=True,
                    do_augment=True),
                CelebD2CAttrFewshotDataset(
                    cls_name=self.conf.manipulate_cls,
                    K=self.conf.manipulate_shots,
                    img_folder=img_folder,
                    img_size=self.conf.img_size,
                    only_cls_name=self.conf.manipulate_cls,
                    only_cls_value=-1,
                    seed=self.conf.manipulate_seed,
                    all_neg=True,
                    do_augment=True),
            ]
        elif self.conf.manipulate_mode == ManipulateMode.celebahq_all:
            return CelebHQAttrDataset(data_paths['celebahq'],
                                      self.conf.img_size,
                                      data_paths['celebahq_anno'],
                                      do_augment=True)
        else:
            raise NotImplementedError()

    def setup(self, stage=None) -> None:
        if self.conf.seed is not None:
            seed = self.conf.seed * get_world_size() + self.global_rank
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print('local seed:', seed)
        self.train_data = self.load_dataset()
        if self.conf.manipulate_mode.is_fewshot():
            if isinstance(self.train_data, list):
                a, b = self.train_data
                self.train_data = [Repeat(a, max(len(a), len(b))),
                                   Repeat(b, max(len(a), len(b)))]
            else:
                self.train_data = Repeat(self.train_data, 100_000)

    def train_dataloader(self):
        conf = self.conf.clone()
        conf.batch_size = self.batch_size
        if isinstance(self.train_data, list):
            dataloader = []
            for each in self.train_data:
                dataloader.append(conf.make_loader(each, shuffle=True, drop_last=True))
            dataloader = ZipLoader(dataloader)
        else:
            dataloader = conf.make_loader(self.train_data, shuffle=True, drop_last=True)
        return dataloader

    @property
    def batch_size(self):
        ws = get_world_size()
        assert self.conf.batch_size % ws == 0
        return self.conf.batch_size // ws

    ####################################
    # Forward and Training Methods
    ####################################
    def forward(self, x, t=None):
        if self.diffusion_time_dependent:
            if t is None:
                t = torch.zeros(x.size(0), device=x.device)
            t_emb = get_timestep_embedding(t, self.time_embedding_dim)
            t_emb = self.time_embed(t_emb)
            x = torch.cat([x, t_emb], dim=-1)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        self.ema_model: BeatGANsAutoencModel
        if isinstance(batch, tuple):
            a, b = batch
            imgs = torch.cat([a['img'], b['img']])
            labels = torch.cat([a['labels'], b['labels']])
        else:
            imgs = batch['img']
            labels = batch['labels']

        if self.conf.train_mode == TrainMode.manipulate:
            self.ema_model.eval()
            with torch.no_grad():
                cond = self.ema_model.encoder(imgs)
            if self.conf.manipulate_znormalize:
                cond = self.normalize(cond)
            if self.diffusion_time_dependent:
                # Sample a random diffusion time between 0 and max_diffusion_time for each example.
                t_rand = torch.randint(0, self.max_diffusion_time + 1, (cond.size(0),), device=cond.device)
                # Instead of creating a new sampler every step, use precomputed alpha and sigma arrays:
                # (Make sure the alpha_vals and sigma_vals tensors are on the same device as cond.)
                alpha_vals = self.alpha_vals.to(cond.device).to(cond.dtype)
                sigma_vals = self.sigma_vals.to(cond.device).to(cond.dtype)
                alpha_t = alpha_vals[t_rand.long()].view(t_rand.size(0), 1)
                sigma_t = sigma_vals[t_rand.long()].view(t_rand.size(0), 1)
                noise = torch.randn_like(cond)
                cond_perturbed = alpha_t * cond + sigma_t * noise
                pred = self.classifier.forward(cond_perturbed, t=t_rand)
                pred_ema = self.ema_classifier.forward(cond_perturbed, t=t_rand)
            else:
                pred = self.classifier.forward(cond)
                pred_ema = self.ema_classifier.forward(cond)
        elif self.conf.train_mode == TrainMode.manipulate_img:
            pred = self.classifier.forward(imgs)
            pred_ema = None
        elif self.conf.train_mode == TrainMode.manipulate_imgt:
            t, weight = self.T_sampler.sample(len(imgs), imgs.device)
            imgs_t = self.sampler.q_sample(imgs, t)
            pred = self.classifier.forward(imgs_t, t=t)
            pred_ema = None
            print('pred:', pred.shape)
        else:
            raise NotImplementedError()

        if self.conf.manipulate_mode.is_celeba_attr():
            gt = torch.where(labels > 0, torch.ones_like(labels).float(), torch.zeros_like(labels).float())
        elif self.conf.manipulate_mode == ManipulateMode.relighting:
            gt = labels
        else:
            raise NotImplementedError()

        if self.conf.manipulate_loss == ManipulateLossType.bce:
            loss = F.binary_cross_entropy_with_logits(pred, gt)
            if pred_ema is not None:
                loss_ema = F.binary_cross_entropy_with_logits(pred_ema, gt)
        elif self.conf.manipulate_loss == ManipulateLossType.mse:
            loss = F.mse_loss(pred, gt)
            if pred_ema is not None:
                loss_ema = F.mse_loss(pred_ema, gt)
        else:
            raise NotImplementedError()

        self.log('loss', loss)
        self.log('loss_ema', loss_ema)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        ema(self.classifier, self.ema_classifier, self.conf.ema_decay)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.classifier.parameters(),
                                 lr=self.conf.lr,
                                 weight_decay=self.conf.weight_decay)
        return optim

####################################
# EMA Helper and Training Function
####################################
def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(target_dict[key].data * decay +
                                    source_dict[key].data * (1 - decay))

def train_cls(conf: TrainConfig, gpus):
    print('conf:', conf.name)
    model = ClsModel(conf)
    if not os.path.exists(conf.logdir):
        os.makedirs(conf.logdir)
    checkpoint = ModelCheckpoint(
        dirpath=f'{conf.logdir}',
        save_last=True,
        save_top_k=1,
    )
    checkpoint_path = f'{conf.logdir}/last.ckpt'
    if os.path.exists(checkpoint_path):
        resume = checkpoint_path
    else:
        if conf.continue_from is not None:
            resume = conf.continue_from.path
        else:
            resume = None
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=conf.logdir,
        name=None,
        version=''
    )
    plugins = []
    accelerator = "gpu"
    if len(gpus) > 1:
        from pytorch_lightning.plugins import DDPPlugin
        plugins.append(DDPPlugin(find_unused_parameters=False))
    trainer = pl.Trainer(
        max_steps=60000,
        devices=gpus,
        accelerator=accelerator,
        precision=16 if conf.fp16 else 32,
        callbacks=[checkpoint],
        logger=tb_logger,
        accumulate_grad_batches=conf.accum_batches,
        plugins=plugins,
    )
    trainer.fit(model, ckpt_path=resume)
