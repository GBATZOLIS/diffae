
# FILE: ./run_ffhq256_latent.py

from templates import *
from templates_latent import *

if __name__ == '__main__':
    # do run the run_ffhq256 before using the file to train the latent DPM

    # infer the latents for training the latent DPM
    # NOTE: not gpu heavy, but more gpus can be of use!
    gpus = [0, 1, 2, 3]
    conf = ffhq256_autoenc()
    conf.eval_programs = ['infer']
    train(conf, gpus=gpus, mode='eval')

    # train the latent DPM
    # NOTE: only need a single gpu
    gpus = [0]
    conf = ffhq256_autoenc_latent()
    train(conf, gpus=gpus)

# FILE: ./metrics.py

import os
import shutil

import torch
import torchvision
from pytorch_fid import fid_score
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.autonotebook import tqdm, trange

from renderer import *
from config import *
from diffusion import Sampler
from dist_utils import *
import lpips
from ssim import ssim


def make_subset_loader(conf: TrainConfig,
                       dataset: Dataset,
                       batch_size: int,
                       shuffle: bool,
                       parallel: bool,
                       drop_last=True):
    dataset = SubsetDataset(dataset, size=conf.eval_num_images)
    if parallel and distributed.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    else:
        sampler = None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        # with sampler, use the sample instead of this option
        shuffle=False if sampler else shuffle,
        num_workers=conf.num_workers,
        pin_memory=True,
        drop_last=drop_last,
        multiprocessing_context=get_context('fork'),
    )


def evaluate_lpips(
    sampler: Sampler,
    model: Model,
    conf: TrainConfig,
    device,
    val_data: Dataset,
    latent_sampler: Sampler = None,
    use_inverted_noise: bool = False,
):
    """
    compare the generated images from autoencoder on validation dataset

    Args:
        use_inversed_noise: the noise is also inverted from DDIM
    """
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    val_loader = make_subset_loader(conf,
                                    dataset=val_data,
                                    batch_size=conf.batch_size_eval,
                                    shuffle=False,
                                    parallel=True)

    model.eval()
    with torch.no_grad():
        scores = {
            'lpips': [],
            'mse': [],
            'ssim': [],
            'psnr': [],
        }
        for batch in tqdm(val_loader, desc='lpips'):
            imgs = batch['img'].to(device)

            if use_inverted_noise:
                # inverse the noise
                # with condition from the encoder
                model_kwargs = {}
                if conf.model_type.has_autoenc():
                    with torch.no_grad():
                        model_kwargs = model.encode(imgs)
                x_T = sampler.ddim_reverse_sample_loop(
                    model=model,
                    x=imgs,
                    clip_denoised=True,
                    model_kwargs=model_kwargs)
                x_T = x_T['sample']
            else:
                x_T = torch.randn((len(imgs), 3, conf.img_size, conf.img_size),
                                  device=device)

            if conf.model_type == ModelType.ddpm:
                # the case where you want to calculate the inversion capability of the DDIM model
                assert use_inverted_noise
                pred_imgs = render_uncondition(
                    conf=conf,
                    model=model,
                    x_T=x_T,
                    sampler=sampler,
                    latent_sampler=latent_sampler,
                )
            else:
                pred_imgs = render_condition(conf=conf,
                                             model=model,
                                             x_T=x_T,
                                             x_start=imgs,
                                             cond=None,
                                             sampler=sampler)
            # # returns {'cond', 'cond2'}
            # conds = model.encode(imgs)
            # pred_imgs = sampler.sample(model=model,
            #                            noise=x_T,
            #                            model_kwargs=conds)

            # (n, 1, 1, 1) => (n, )
            scores['lpips'].append(lpips_fn.forward(imgs, pred_imgs).view(-1))

            # need to normalize into [0, 1]
            norm_imgs = (imgs + 1) / 2
            norm_pred_imgs = (pred_imgs + 1) / 2
            # (n, )
            scores['ssim'].append(
                ssim(norm_imgs, norm_pred_imgs, size_average=False))
            # (n, )
            scores['mse'].append(
                (norm_imgs - norm_pred_imgs).pow(2).mean(dim=[1, 2, 3]))
            # (n, )
            scores['psnr'].append(psnr(norm_imgs, norm_pred_imgs))
        # (N, )
        for key in scores.keys():
            scores[key] = torch.cat(scores[key]).float()
    model.train()

    barrier()

    # support multi-gpu
    outs = {
        key: [
            torch.zeros(len(scores[key]), device=device)
            for i in range(get_world_size())
        ]
        for key in scores.keys()
    }
    for key in scores.keys():
        all_gather(outs[key], scores[key])

    # final scores
    for key in scores.keys():
        scores[key] = torch.cat(outs[key]).mean().item()

    # {'lpips', 'mse', 'ssim'}
    return scores


def psnr(img1, img2):
    """
    Args:
        img1: (n, c, h, w)
    """
    v_max = 1.
    # (n,)
    mse = torch.mean((img1 - img2)**2, dim=[1, 2, 3])
    return 20 * torch.log10(v_max / torch.sqrt(mse))


def evaluate_fid(
    sampler: Sampler,
    model: Model,
    conf: TrainConfig,
    device,
    train_data: Dataset,
    val_data: Dataset,
    latent_sampler: Sampler = None,
    conds_mean=None,
    conds_std=None,
    remove_cache: bool = True,
    clip_latent_noise: bool = False,
):
    assert conf.fid_cache is not None
    if get_rank() == 0:
        # no parallel
        # validation data for a comparing FID
        val_loader = make_subset_loader(conf,
                                        dataset=val_data,
                                        batch_size=conf.batch_size_eval,
                                        shuffle=False,
                                        parallel=False)

        # put the val images to a directory
        cache_dir = f'{conf.fid_cache}_{conf.eval_num_images}'
        if (os.path.exists(cache_dir)
                and len(os.listdir(cache_dir)) < conf.eval_num_images):
            shutil.rmtree(cache_dir)

        if not os.path.exists(cache_dir):
            # write files to the cache
            # the images are normalized, hence need to denormalize first
            loader_to_path(val_loader, cache_dir, denormalize=True)

        # create the generate dir
        if os.path.exists(conf.generate_dir):
            shutil.rmtree(conf.generate_dir)
        os.makedirs(conf.generate_dir)

    barrier()

    world_size = get_world_size()
    rank = get_rank()
    batch_size = chunk_size(conf.batch_size_eval, rank, world_size)

    def filename(idx):
        return world_size * idx + rank

    model.eval()
    with torch.no_grad():
        if conf.model_type.can_sample():
            eval_num_images = chunk_size(conf.eval_num_images, rank,
                                         world_size)
            desc = "generating images"
            for i in trange(0, eval_num_images, batch_size, desc=desc):
                batch_size = min(batch_size, eval_num_images - i)
                x_T = torch.randn(
                    (batch_size, 3, conf.img_size, conf.img_size),
                    device=device)
                batch_images = render_uncondition(
                    conf=conf,
                    model=model,
                    x_T=x_T,
                    sampler=sampler,
                    latent_sampler=latent_sampler,
                    conds_mean=conds_mean,
                    conds_std=conds_std).cpu()

                batch_images = (batch_images + 1) / 2
                # keep the generated images
                for j in range(len(batch_images)):
                    img_name = filename(i + j)
                    torchvision.utils.save_image(
                        batch_images[j],
                        os.path.join(conf.generate_dir, f'{img_name}.png'))
        elif conf.model_type == ModelType.autoencoder:
            if conf.train_mode.is_latent_diffusion():
                # evaluate autoencoder + latent diffusion (doesn't give the images)
                model: BeatGANsAutoencModel
                eval_num_images = chunk_size(conf.eval_num_images, rank,
                                             world_size)
                desc = "generating images"
                for i in trange(0, eval_num_images, batch_size, desc=desc):
                    batch_size = min(batch_size, eval_num_images - i)
                    x_T = torch.randn(
                        (batch_size, 3, conf.img_size, conf.img_size),
                        device=device)
                    batch_images = render_uncondition(
                        conf=conf,
                        model=model,
                        x_T=x_T,
                        sampler=sampler,
                        latent_sampler=latent_sampler,
                        conds_mean=conds_mean,
                        conds_std=conds_std,
                        clip_latent_noise=clip_latent_noise,
                    ).cpu()
                    batch_images = (batch_images + 1) / 2
                    # keep the generated images
                    for j in range(len(batch_images)):
                        img_name = filename(i + j)
                        torchvision.utils.save_image(
                            batch_images[j],
                            os.path.join(conf.generate_dir, f'{img_name}.png'))
            else:
                # evaulate autoencoder (given the images)
                # to make the FID fair, autoencoder must not see the validation dataset
                # also shuffle to make it closer to unconditional generation
                train_loader = make_subset_loader(conf,
                                                  dataset=train_data,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  parallel=True)

                i = 0
                for batch in tqdm(train_loader, desc='generating images'):
                    imgs = batch['img'].to(device)
                    x_T = torch.randn(
                        (len(imgs), 3, conf.img_size, conf.img_size),
                        device=device)
                    batch_images = render_condition(
                        conf=conf,
                        model=model,
                        x_T=x_T,
                        x_start=imgs,
                        cond=None,
                        sampler=sampler,
                        latent_sampler=latent_sampler).cpu()
                    # model: BeatGANsAutoencModel
                    # # returns {'cond', 'cond2'}
                    # conds = model.encode(imgs)
                    # batch_images = sampler.sample(model=model,
                    #                               noise=x_T,
                    #                               model_kwargs=conds).cpu()
                    # denormalize the images
                    batch_images = (batch_images + 1) / 2
                    # keep the generated images
                    for j in range(len(batch_images)):
                        img_name = filename(i + j)
                        torchvision.utils.save_image(
                            batch_images[j],
                            os.path.join(conf.generate_dir, f'{img_name}.png'))
                    i += len(imgs)
        else:
            raise NotImplementedError()
    model.train()

    barrier()

    if get_rank() == 0:
        fid = fid_score.calculate_fid_given_paths(
            [cache_dir, conf.generate_dir],
            batch_size,
            device=device,
            dims=2048)

        # remove the cache
        if remove_cache and os.path.exists(conf.generate_dir):
            shutil.rmtree(conf.generate_dir)

    barrier()

    if get_rank() == 0:
        # need to float it! unless the broadcasted value is wrong
        fid = torch.tensor(float(fid), device=device)
        broadcast(fid, 0)
    else:
        fid = torch.tensor(0., device=device)
        broadcast(fid, 0)
    fid = fid.item()
    print(f'fid ({get_rank()}):', fid)

    return fid


def loader_to_path(loader: DataLoader, path: str, denormalize: bool):
    # not process safe!

    if not os.path.exists(path):
        os.makedirs(path)

    # write the loader to files
    i = 0
    for batch in tqdm(loader, desc='copy images'):
        imgs = batch['img']
        if denormalize:
            imgs = (imgs + 1) / 2
        for j in range(len(imgs)):
            torchvision.utils.save_image(imgs[j],
                                         os.path.join(path, f'{i+j}.png'))
        i += len(imgs)

# FILE: ./data_resize_bedroom.py

import argparse
import multiprocessing
import os
from os.path import join, exists
from functools import partial
from io import BytesIO
import shutil

import lmdb
from PIL import Image
from torchvision.datasets import LSUNClass
from torchvision.transforms import functional as trans_fn
from tqdm import tqdm

from multiprocessing import Process, Queue


def resize_and_convert(img, size, resample, quality=100):
    img = trans_fn.resize(img, size, resample)
    img = trans_fn.center_crop(img, size)
    buffer = BytesIO()
    img.save(buffer, format="webp", quality=quality)
    val = buffer.getvalue()

    return val


def resize_multiple(img,
                    sizes=(128, 256, 512, 1024),
                    resample=Image.LANCZOS,
                    quality=100):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size, resample, quality))

    return imgs


def resize_worker(idx, img, sizes, resample):
    img = img.convert("RGB")
    out = resize_multiple(img, sizes=sizes, resample=resample)
    return idx, out


from torch.utils.data import Dataset, DataLoader


class ConvertDataset(Dataset):
    def __init__(self, data) -> None:
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, _ = self.data[index]
        bytes = resize_and_convert(img, 256, Image.LANCZOS, quality=90)
        return bytes


if __name__ == "__main__":
    """
    converting lsun' original lmdb to our lmdb, which is somehow more performant.
    """
    from tqdm import tqdm

    # path to the original lsun's lmdb
    src_path = 'datasets/bedroom_train_lmdb'
    out_path = 'datasets/bedroom256.lmdb'

    dataset = LSUNClass(root=os.path.expanduser(src_path))
    dataset = ConvertDataset(dataset)
    loader = DataLoader(dataset,
                        batch_size=50,
                        num_workers=12,
                        collate_fn=lambda x: x,
                        shuffle=False)

    target = os.path.expanduser(out_path)
    if os.path.exists(target):
        shutil.rmtree(target)

    with lmdb.open(target, map_size=1024**4, readahead=False) as env:
        with tqdm(total=len(dataset)) as progress:
            i = 0
            for batch in loader:
                with env.begin(write=True) as txn:
                    for img in batch:
                        key = f"{256}-{str(i).zfill(7)}".encode("utf-8")
                        # print(key)
                        txn.put(key, img)
                        i += 1
                        progress.update()
                # if i == 1000:
                #     break
                # if total == len(imgset):
                #     break

        with env.begin(write=True) as txn:
            txn.put("length".encode("utf-8"), str(i).encode("utf-8"))

# FILE: ./run_ffhq128_ddim.py

from templates import *
from templates_latent import *

if __name__ == '__main__':
    gpus = [0, 1, 2, 3]
    conf = ffhq128_ddpm_130M()
    train(conf, gpus=gpus)

    gpus = [0, 1, 2, 3]
    conf.eval_programs = ['fid10']
    train(conf, gpus=gpus, mode='eval')
# FILE: ./config.py

from model.unet import ScaleAt
from model.latentnet import *
from diffusion.resample import UniformSampler
from diffusion.diffusion import space_timesteps
from typing import Tuple

from torch.utils.data import DataLoader

from config_base import BaseConfig
from dataset import *
from diffusion import *
from diffusion.base import GenerativeType, LossType, ModelMeanType, ModelVarType, get_named_beta_schedule
from model import *
from choices import *
from multiprocessing import get_context
import os
from dataset_util import *
from torch.utils.data.distributed import DistributedSampler

data_paths = {
    'ffhqlmdb256':
    os.path.expanduser('datasets/ffhq256.lmdb'),
    # used for training a classifier
    'celeba':
    os.path.expanduser('datasets/celeba'),
    # used for training DPM models
    'celebalmdb':
    os.path.expanduser('datasets/celeba.lmdb'),
    'celebahq':
    os.path.expanduser('datasets/celebahq256.lmdb'),
    'horse256':
    os.path.expanduser('datasets/horse256.lmdb'),
    'bedroom256':
    os.path.expanduser('datasets/bedroom256.lmdb'),
    'celeba_anno':
    os.path.expanduser('datasets/celeba_anno/list_attr_celeba.txt'),
    'celebahq_anno':
    os.path.expanduser(
        'datasets/celeba_anno/CelebAMask-HQ-attribute-anno.txt'),
    'celeba_relight':
    os.path.expanduser('datasets/celeba_hq_light/celeba_light.txt'),
}


@dataclass
class PretrainConfig(BaseConfig):
    name: str
    path: str


@dataclass
class TrainConfig(BaseConfig):
    # random seed
    seed: int = 0
    train_mode: TrainMode = TrainMode.diffusion
    train_cond0_prob: float = 0
    train_pred_xstart_detach: bool = True
    train_interpolate_prob: float = 0
    train_interpolate_img: bool = False
    manipulate_mode: ManipulateMode = ManipulateMode.celebahq_all
    manipulate_cls: str = None
    manipulate_shots: int = None
    manipulate_loss: ManipulateLossType = ManipulateLossType.bce
    manipulate_znormalize: bool = False
    manipulate_seed: int = 0
    accum_batches: int = 1
    autoenc_mid_attn: bool = True
    batch_size: int = 16
    batch_size_eval: int = None
    beatgans_gen_type: GenerativeType = GenerativeType.ddim
    beatgans_loss_type: LossType = LossType.mse
    beatgans_model_mean_type: ModelMeanType = ModelMeanType.eps
    beatgans_model_var_type: ModelVarType = ModelVarType.fixed_large
    beatgans_rescale_timesteps: bool = False
    latent_infer_path: str = None
    latent_znormalize: bool = False
    latent_gen_type: GenerativeType = GenerativeType.ddim
    latent_loss_type: LossType = LossType.mse
    latent_model_mean_type: ModelMeanType = ModelMeanType.eps
    latent_model_var_type: ModelVarType = ModelVarType.fixed_large
    latent_rescale_timesteps: bool = False
    latent_T_eval: int = 1_000
    latent_clip_sample: bool = False
    latent_beta_scheduler: str = 'linear'
    beta_scheduler: str = 'linear'
    data_name: str = ''
    data_val_name: str = None
    diffusion_type: str = None
    dropout: float = 0.1
    ema_decay: float = 0.9999
    eval_num_images: int = 5_000
    eval_every_samples: int = 200_000
    eval_ema_every_samples: int = 200_000
    fid_use_torch: bool = True
    fp16: bool = False
    grad_clip: float = 1
    img_size: int = 64
    lr: float = 0.0001
    optimizer: OptimizerType = OptimizerType.adam
    weight_decay: float = 0
    model_conf: ModelConfig = None
    model_name: ModelName = None
    model_type: ModelType = None
    net_attn: Tuple[int] = None
    net_beatgans_attn_head: int = 1
    # not necessarily the same as the the number of style channels
    net_beatgans_embed_channels: int = 512
    net_resblock_updown: bool = True
    net_enc_use_time: bool = False
    net_enc_pool: str = 'adaptivenonzero'
    net_beatgans_gradient_checkpoint: bool = False
    net_beatgans_resnet_two_cond: bool = False
    net_beatgans_resnet_use_zero_module: bool = True
    net_beatgans_resnet_scale_at: ScaleAt = ScaleAt.after_norm
    net_beatgans_resnet_cond_channels: int = None
    net_ch_mult: Tuple[int] = None
    net_ch: int = 64
    net_enc_attn: Tuple[int] = None
    net_enc_k: int = None
    # number of resblocks for the encoder (half-unet)
    net_enc_num_res_blocks: int = 2
    net_enc_channel_mult: Tuple[int] = None
    net_enc_grad_checkpoint: bool = False
    net_autoenc_stochastic: bool = False
    net_latent_activation: Activation = Activation.silu
    net_latent_channel_mult: Tuple[int] = (1, 2, 4)
    net_latent_condition_bias: float = 0
    net_latent_dropout: float = 0
    net_latent_layers: int = None
    net_latent_net_last_act: Activation = Activation.none
    net_latent_net_type: LatentNetType = LatentNetType.none
    net_latent_num_hid_channels: int = 1024
    net_latent_num_time_layers: int = 2
    net_latent_skip_layers: Tuple[int] = None
    net_latent_time_emb_channels: int = 64
    net_latent_use_norm: bool = False
    net_latent_time_last_act: bool = False
    net_num_res_blocks: int = 2
    # number of resblocks for the UNET
    net_num_input_res_blocks: int = None
    net_enc_num_cls: int = None
    num_workers: int = 4
    parallel: bool = False
    postfix: str = ''
    sample_size: int = 64
    sample_every_samples: int = 20_000
    save_every_samples: int = 100_000
    style_ch: int = 512
    T_eval: int = 1_000
    T_sampler: str = 'uniform'
    T: int = 1_000
    total_samples: int = 10_000_000
    warmup: int = 0
    pretrain: PretrainConfig = None
    continue_from: PretrainConfig = None
    eval_programs: Tuple[str] = None
    # if present load the checkpoint from this path instead
    eval_path: str = None
    base_dir: str = 'checkpoints'
    use_cache_dataset: bool = False
    data_cache_dir: str = os.path.expanduser('~/cache')
    work_cache_dir: str = os.path.expanduser('~/mycache')
    # to be overridden
    name: str = ''

    def __post_init__(self):
        self.batch_size_eval = self.batch_size_eval or self.batch_size
        self.data_val_name = self.data_val_name or self.data_name

    def scale_up_gpus(self, num_gpus, num_nodes=1):
        self.eval_ema_every_samples *= num_gpus * num_nodes
        self.eval_every_samples *= num_gpus * num_nodes
        self.sample_every_samples *= num_gpus * num_nodes
        self.batch_size *= num_gpus * num_nodes
        self.batch_size_eval *= num_gpus * num_nodes
        return self

    @property
    def batch_size_effective(self):
        return self.batch_size * self.accum_batches

    @property
    def fid_cache(self):
        # we try to use the local dirs to reduce the load over network drives
        # hopefully, this would reduce the disconnection problems with sshfs
        return f'{self.work_cache_dir}/eval_images/{self.data_name}_size{self.img_size}_{self.eval_num_images}'

    @property
    def data_path(self):
        # may use the cache dir
        path = data_paths[self.data_name]
        if self.use_cache_dataset and path is not None:
            path = use_cached_dataset_path(
                path, f'{self.data_cache_dir}/{self.data_name}')
        return path

    @property
    def logdir(self):
        return f'{self.base_dir}/{self.name}'

    @property
    def generate_dir(self):
        # we try to use the local dirs to reduce the load over network drives
        # hopefully, this would reduce the disconnection problems with sshfs
        return f'{self.work_cache_dir}/gen_images/{self.name}'

    def _make_diffusion_conf(self, T=None):
        if self.diffusion_type == 'beatgans':
            # can use T < self.T for evaluation
            # follows the guided-diffusion repo conventions
            # t's are evenly spaced
            if self.beatgans_gen_type == GenerativeType.ddpm:
                section_counts = [T]
            elif self.beatgans_gen_type == GenerativeType.ddim:
                section_counts = f'ddim{T}'
            else:
                raise NotImplementedError()

            return SpacedDiffusionBeatGansConfig(
                gen_type=self.beatgans_gen_type,
                model_type=self.model_type,
                betas=get_named_beta_schedule(self.beta_scheduler, self.T),
                model_mean_type=self.beatgans_model_mean_type,
                model_var_type=self.beatgans_model_var_type,
                loss_type=self.beatgans_loss_type,
                rescale_timesteps=self.beatgans_rescale_timesteps,
                use_timesteps=space_timesteps(num_timesteps=self.T,
                                              section_counts=section_counts),
                fp16=self.fp16,
            )
        else:
            raise NotImplementedError()

    def _make_latent_diffusion_conf(self, T=None):
        # can use T < self.T for evaluation
        # follows the guided-diffusion repo conventions
        # t's are evenly spaced
        if self.latent_gen_type == GenerativeType.ddpm:
            section_counts = [T]
        elif self.latent_gen_type == GenerativeType.ddim:
            section_counts = f'ddim{T}'
        else:
            raise NotImplementedError()

        return SpacedDiffusionBeatGansConfig(
            train_pred_xstart_detach=self.train_pred_xstart_detach,
            gen_type=self.latent_gen_type,
            # latent's model is always ddpm
            model_type=ModelType.ddpm,
            # latent shares the beta scheduler and full T
            betas=get_named_beta_schedule(self.latent_beta_scheduler, self.T),
            model_mean_type=self.latent_model_mean_type,
            model_var_type=self.latent_model_var_type,
            loss_type=self.latent_loss_type,
            rescale_timesteps=self.latent_rescale_timesteps,
            use_timesteps=space_timesteps(num_timesteps=self.T,
                                          section_counts=section_counts),
            fp16=self.fp16,
        )

    @property
    def model_out_channels(self):
        return 3

    def make_T_sampler(self):
        if self.T_sampler == 'uniform':
            return UniformSampler(self.T)
        else:
            raise NotImplementedError()

    def make_diffusion_conf(self):
        return self._make_diffusion_conf(self.T)

    def make_eval_diffusion_conf(self):
        return self._make_diffusion_conf(T=self.T_eval)

    def make_latent_diffusion_conf(self):
        return self._make_latent_diffusion_conf(T=self.T)

    def make_latent_eval_diffusion_conf(self):
        # latent can have different eval T
        return self._make_latent_diffusion_conf(T=self.latent_T_eval)

    def make_dataset(self, path=None, **kwargs):
        if self.data_name == 'ffhqlmdb256':
            return FFHQlmdb(path=path or self.data_path,
                            image_size=self.img_size,
                            **kwargs)
        elif self.data_name == 'horse256':
            return Horse_lmdb(path=path or self.data_path,
                              image_size=self.img_size,
                              **kwargs)
        elif self.data_name == 'bedroom256':
            return Horse_lmdb(path=path or self.data_path,
                              image_size=self.img_size,
                              **kwargs)
        elif self.data_name == 'celebalmdb':
            # always use d2c crop
            return CelebAlmdb(path=path or self.data_path,
                              image_size=self.img_size,
                              original_resolution=None,
                              crop_d2c=True,
                              **kwargs)
        else:
            raise NotImplementedError()

    def make_loader(self,
                    dataset,
                    shuffle: bool,
                    num_worker: bool = None,
                    drop_last: bool = True,
                    batch_size: int = None,
                    parallel: bool = False):
        if parallel and distributed.is_initialized():
            # drop last to make sure that there is no added special indexes
            sampler = DistributedSampler(dataset,
                                         shuffle=shuffle,
                                         drop_last=True)
        else:
            sampler = None
        return DataLoader(
            dataset,
            batch_size=batch_size or self.batch_size,
            sampler=sampler,
            # with sampler, use the sample instead of this option
            shuffle=False if sampler else shuffle,
            num_workers=num_worker or self.num_workers,
            pin_memory=True,
            drop_last=drop_last,
            multiprocessing_context=get_context('fork'),
        )

    def make_model_conf(self):
        if self.model_name == ModelName.beatgans_ddpm:
            self.model_type = ModelType.ddpm
            self.model_conf = BeatGANsUNetConfig(
                attention_resolutions=self.net_attn,
                channel_mult=self.net_ch_mult,
                conv_resample=True,
                dims=2,
                dropout=self.dropout,
                embed_channels=self.net_beatgans_embed_channels,
                image_size=self.img_size,
                in_channels=3,
                model_channels=self.net_ch,
                num_classes=None,
                num_head_channels=-1,
                num_heads_upsample=-1,
                num_heads=self.net_beatgans_attn_head,
                num_res_blocks=self.net_num_res_blocks,
                num_input_res_blocks=self.net_num_input_res_blocks,
                out_channels=self.model_out_channels,
                resblock_updown=self.net_resblock_updown,
                use_checkpoint=self.net_beatgans_gradient_checkpoint,
                use_new_attention_order=False,
                resnet_two_cond=self.net_beatgans_resnet_two_cond,
                resnet_use_zero_module=self.
                net_beatgans_resnet_use_zero_module,
            )
        elif self.model_name in [
                ModelName.beatgans_autoenc,
        ]:
            cls = BeatGANsAutoencConfig
            # supports both autoenc and vaeddpm
            if self.model_name == ModelName.beatgans_autoenc:
                self.model_type = ModelType.autoencoder
            else:
                raise NotImplementedError()

            if self.net_latent_net_type == LatentNetType.none:
                latent_net_conf = None
            elif self.net_latent_net_type == LatentNetType.skip:
                latent_net_conf = MLPSkipNetConfig(
                    num_channels=self.style_ch,
                    skip_layers=self.net_latent_skip_layers,
                    num_hid_channels=self.net_latent_num_hid_channels,
                    num_layers=self.net_latent_layers,
                    num_time_emb_channels=self.net_latent_time_emb_channels,
                    activation=self.net_latent_activation,
                    use_norm=self.net_latent_use_norm,
                    condition_bias=self.net_latent_condition_bias,
                    dropout=self.net_latent_dropout,
                    last_act=self.net_latent_net_last_act,
                    num_time_layers=self.net_latent_num_time_layers,
                    time_last_act=self.net_latent_time_last_act,
                )
            else:
                raise NotImplementedError()

            self.model_conf = cls(
                attention_resolutions=self.net_attn,
                channel_mult=self.net_ch_mult,
                conv_resample=True,
                dims=2,
                dropout=self.dropout,
                embed_channels=self.net_beatgans_embed_channels,
                enc_out_channels=self.style_ch,
                enc_pool=self.net_enc_pool,
                enc_num_res_block=self.net_enc_num_res_blocks,
                enc_channel_mult=self.net_enc_channel_mult,
                enc_grad_checkpoint=self.net_enc_grad_checkpoint,
                enc_attn_resolutions=self.net_enc_attn,
                image_size=self.img_size,
                in_channels=3,
                model_channels=self.net_ch,
                num_classes=None,
                num_head_channels=-1,
                num_heads_upsample=-1,
                num_heads=self.net_beatgans_attn_head,
                num_res_blocks=self.net_num_res_blocks,
                num_input_res_blocks=self.net_num_input_res_blocks,
                out_channels=self.model_out_channels,
                resblock_updown=self.net_resblock_updown,
                use_checkpoint=self.net_beatgans_gradient_checkpoint,
                use_new_attention_order=False,
                resnet_two_cond=self.net_beatgans_resnet_two_cond,
                resnet_use_zero_module=self.
                net_beatgans_resnet_use_zero_module,
                latent_net_conf=latent_net_conf,
                resnet_cond_channels=self.net_beatgans_resnet_cond_channels,
            )
        else:
            raise NotImplementedError(self.model_name)

        return self.model_conf

# FILE: ./align.py

import bz2
import os
import os.path as osp
import sys
from multiprocessing import Pool

import dlib
import numpy as np
import PIL.Image
import requests
import scipy.ndimage
from tqdm import tqdm
from argparse import ArgumentParser

LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'


def image_align(src_file,
                dst_file,
                face_landmarks,
                output_size=1024,
                transform_size=4096,
                enable_padding=True):
    # Align function from FFHQ dataset pre-processing step
    # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

    lm = np.array(face_landmarks)
    lm_chin = lm[0:17]  # left-right
    lm_eyebrow_left = lm[17:22]  # left-right
    lm_eyebrow_right = lm[22:27]  # left-right
    lm_nose = lm[27:31]  # top-down
    lm_nostrils = lm[31:36]  # top-down
    lm_eye_left = lm[36:42]  # left-clockwise
    lm_eye_right = lm[42:48]  # left-clockwise
    lm_mouth_outer = lm[48:60]  # left-clockwise
    lm_mouth_inner = lm[60:68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Load in-the-wild image.
    if not os.path.isfile(src_file):
        print(
            '\nCannot find source image. Please run "--wilds" before "--align".'
        )
        return
    img = PIL.Image.open(src_file)
    img = img.convert('RGB')

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)),
                 int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0),
            min(crop[2] + border,
                img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
           int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border,
               0), max(-pad[1] + border,
                       0), max(pad[2] - img.size[0] + border,
                               0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img),
                     ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(
            1.0 -
            np.minimum(np.float32(x) / pad[0],
                       np.float32(w - 1 - x) / pad[2]), 1.0 -
            np.minimum(np.float32(y) / pad[1],
                       np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) -
                img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)),
                                  'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD,
                        (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    # Save aligned image.
    img.save(dst_file, 'PNG')


class LandmarksDetector:
    def __init__(self, predictor_model_path):
        """
        :param predictor_model_path: path to shape_predictor_68_face_landmarks.dat file
        """
        self.detector = dlib.get_frontal_face_detector(
        )  # cnn_face_detection_model_v1 also can be used
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)

    def get_landmarks(self, image):
        img = dlib.load_rgb_image(image)
        dets = self.detector(img, 1)

        for detection in dets:
            face_landmarks = [
                (item.x, item.y)
                for item in self.shape_predictor(img, detection).parts()
            ]
            yield face_landmarks


def unpack_bz2(src_path):
    dst_path = src_path[:-4]
    if os.path.exists(dst_path):
        print('cached')
        return dst_path
    data = bz2.BZ2File(src_path).read()
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path


def work_landmark(raw_img_path, img_name, face_landmarks):
    face_img_name = '%s.png' % (os.path.splitext(img_name)[0], )
    aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)
    if os.path.exists(aligned_face_path):
        return
    image_align(raw_img_path,
                aligned_face_path,
                face_landmarks,
                output_size=256)


def get_file(src, tgt):
    if os.path.exists(tgt):
        print('cached')
        return tgt
    tgt_dir = os.path.dirname(tgt)
    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)
    file = requests.get(src)
    open(tgt, 'wb').write(file.content)
    return tgt


if __name__ == "__main__":
    """
    Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step
    python align_images.py /raw_images /aligned_images
    """
    parser = ArgumentParser()
    parser.add_argument("-i",
                        "--input_imgs_path",
                        type=str,
                        default="imgs",
                        help="input images directory path")
    parser.add_argument("-o",
                        "--output_imgs_path",
                        type=str,
                        default="imgs_align",
                        help="output images directory path")

    args = parser.parse_args()

    # takes very long time  ...
    landmarks_model_path = unpack_bz2(
        get_file(
            'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2',
            'temp/shape_predictor_68_face_landmarks.dat.bz2'))

    # RAW_IMAGES_DIR = sys.argv[1]
    # ALIGNED_IMAGES_DIR = sys.argv[2]
    RAW_IMAGES_DIR = args.input_imgs_path
    ALIGNED_IMAGES_DIR = args.output_imgs_path

    if not osp.exists(ALIGNED_IMAGES_DIR): os.makedirs(ALIGNED_IMAGES_DIR)

    files = os.listdir(RAW_IMAGES_DIR)
    print(f'total img files {len(files)}')
    with tqdm(total=len(files)) as progress:

        def cb(*args):
            # print('update')
            progress.update()

        def err_cb(e):
            print('error:', e)

        with Pool(8) as pool:
            res = []
            landmarks_detector = LandmarksDetector(landmarks_model_path)
            for img_name in files:
                raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
                # print('img_name:', img_name)
                for i, face_landmarks in enumerate(
                        landmarks_detector.get_landmarks(raw_img_path),
                        start=1):
                    # assert i == 1, f'{i}'
                    # print(i, face_landmarks)
                    # face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], i)
                    # aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)
                    # image_align(raw_img_path, aligned_face_path, face_landmarks, output_size=256)

                    work_landmark(raw_img_path, img_name, face_landmarks)
                    progress.update()

                    # job = pool.apply_async(
                    #     work_landmark,
                    #     (raw_img_path, img_name, face_landmarks),
                    #     callback=cb,
                    #     error_callback=err_cb,
                    # )
                    # res.append(job)

            # pool.close()
            # pool.join()
    print(f"output aligned images at: {ALIGNED_IMAGES_DIR}")

# FILE: ./ssim.py

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([
        exp(-(x - window_size // 2)**2 / float(2 * sigma**2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(
        img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(
        img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(
        img1 * img2, window, padding=window_size // 2,
        groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type(
        ) == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel,
                     self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)
# FILE: ./run_ffhq256.py

from templates import *
from templates_latent import *

if __name__ == '__main__':
    # 256 requires 8x v100s, in our case, on two nodes.
    # do not run this directly, use `sbatch run_ffhq256.sh` to spawn the srun properly.
    gpus = [0, 1, 2, 3]
    nodes = 2
    conf = ffhq256_autoenc()
    train(conf, gpus=gpus, nodes=nodes)
# FILE: ./renderer.py

from config import *

from torch.cuda import amp


def render_uncondition(conf: TrainConfig,
                       model: BeatGANsAutoencModel,
                       x_T,
                       sampler: Sampler,
                       latent_sampler: Sampler,
                       conds_mean=None,
                       conds_std=None,
                       clip_latent_noise: bool = False):
    device = x_T.device
    if conf.train_mode == TrainMode.diffusion:
        assert conf.model_type.can_sample()
        return sampler.sample(model=model, noise=x_T)
    elif conf.train_mode.is_latent_diffusion():
        model: BeatGANsAutoencModel
        if conf.train_mode == TrainMode.latent_diffusion:
            latent_noise = torch.randn(len(x_T), conf.style_ch, device=device)
        else:
            raise NotImplementedError()

        if clip_latent_noise:
            latent_noise = latent_noise.clip(-1, 1)

        cond = latent_sampler.sample(
            model=model.latent_net,
            noise=latent_noise,
            clip_denoised=conf.latent_clip_sample,
        )

        if conf.latent_znormalize:
            cond = cond * conds_std.to(device) + conds_mean.to(device)

        # the diffusion on the model
        return sampler.sample(model=model, noise=x_T, cond=cond)
    else:
        raise NotImplementedError()


def render_condition(
    conf: TrainConfig,
    model: BeatGANsAutoencModel,
    x_T,
    sampler: Sampler,
    x_start=None,
    cond=None,
):
    print(conf.train_mode)
    if conf.train_mode in [TrainMode.diffusion,TrainMode.latent_diffusion] :
        assert conf.model_type.has_autoenc()
        # returns {'cond', 'cond2'}
        if cond is None:
            cond = model.encode(x_start)
        return sampler.sample(model=model,
                              noise=x_T,
                              model_kwargs={'cond': cond})
    else:
        raise NotImplementedError()

# FILE: ./run_horse128.py

from templates import *
from templates_latent import *

if __name__ == '__main__':
    # train the autoenc moodel
    # this requires V100s.
    gpus = [0, 1, 2, 3]
    conf = horse128_autoenc()
    train(conf, gpus=gpus)

    # infer the latents for training the latent DPM
    # NOTE: not gpu heavy, but more gpus can be of use!
    gpus = [0, 1, 2, 3]
    conf.eval_programs = ['infer']
    train(conf, gpus=gpus, mode='eval')

    # train the latent DPM
    # NOTE: only need a single gpu
    gpus = [0]
    conf = horse128_autoenc_latent()
    train(conf, gpus=gpus)

    # unconditional sampling score
    # NOTE: a lot of gpus can speed up this process
    gpus = [0, 1, 2, 3]
    conf.eval_programs = ['fid(10,10)']
    train(conf, gpus=gpus, mode='eval')
# FILE: ./predict.py

# pre-download the weights for 256 resolution model to checkpoints/ffhq256_autoenc and checkpoints/ffhq256_autoenc_cls
# wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# bunzip2 shape_predictor_68_face_landmarks.dat.bz2

import os
import torch
from torchvision.utils import save_image
import tempfile
from templates import *
from templates_cls import *
from experiment_classifier import ClsModel
from align import LandmarksDetector, image_align
from cog import BasePredictor, Path, Input, BaseModel


class ModelOutput(BaseModel):
    image: Path


class Predictor(BasePredictor):
    def setup(self):
        self.aligned_dir = "aligned"
        os.makedirs(self.aligned_dir, exist_ok=True)
        self.device = "cuda:0"

        # Model Initialization
        model_config = ffhq256_autoenc()
        self.model = LitModel(model_config)
        state = torch.load("checkpoints/ffhq256_autoenc/last.ckpt", map_location="cpu")
        self.model.load_state_dict(state["state_dict"], strict=False)
        self.model.ema_model.eval()
        self.model.ema_model.to(self.device)

        # Classifier Initialization
        classifier_config = ffhq256_autoenc_cls()
        classifier_config.pretrain = None  # a bit faster
        self.classifier = ClsModel(classifier_config)
        state_class = torch.load(
            "checkpoints/ffhq256_autoenc_cls/last.ckpt", map_location="cpu"
        )
        print("latent step:", state_class["global_step"])
        self.classifier.load_state_dict(state_class["state_dict"], strict=False)
        self.classifier.to(self.device)

        self.landmarks_detector = LandmarksDetector(
            "shape_predictor_68_face_landmarks.dat"
        )

    def predict(
        self,
        image: Path = Input(
            description="Input image for face manipulation. Image will be aligned and cropped, "
            "output aligned and manipulated images.",
        ),
        target_class: str = Input(
            default="Bangs",
            choices=[
                "5_o_Clock_Shadow",
                "Arched_Eyebrows",
                "Attractive",
                "Bags_Under_Eyes",
                "Bald",
                "Bangs",
                "Big_Lips",
                "Big_Nose",
                "Black_Hair",
                "Blond_Hair",
                "Blurry",
                "Brown_Hair",
                "Bushy_Eyebrows",
                "Chubby",
                "Double_Chin",
                "Eyeglasses",
                "Goatee",
                "Gray_Hair",
                "Heavy_Makeup",
                "High_Cheekbones",
                "Male",
                "Mouth_Slightly_Open",
                "Mustache",
                "Narrow_Eyes",
                "Beard",
                "Oval_Face",
                "Pale_Skin",
                "Pointy_Nose",
                "Receding_Hairline",
                "Rosy_Cheeks",
                "Sideburns",
                "Smiling",
                "Straight_Hair",
                "Wavy_Hair",
                "Wearing_Earrings",
                "Wearing_Hat",
                "Wearing_Lipstick",
                "Wearing_Necklace",
                "Wearing_Necktie",
                "Young",
            ],
            description="Choose manipulation direction.",
        ),
        manipulation_amplitude: float = Input(
            default=0.3,
            ge=-0.5,
            le=0.5,
            description="When set too strong it would result in artifact as it could dominate the original image information.",
        ),
        T_step: int = Input(
            default=100,
            choices=[50, 100, 125, 200, 250, 500],
            description="Number of step for generation.",
        ),
        T_inv: int = Input(default=200, choices=[50, 100, 125, 200, 250, 500]),
    ) -> List[ModelOutput]:

        img_size = 256
        print("Aligning image...")
        for i, face_landmarks in enumerate(
            self.landmarks_detector.get_landmarks(str(image)), start=1
        ):
            image_align(str(image), f"{self.aligned_dir}/aligned.png", face_landmarks)

        data = ImageDataset(
            self.aligned_dir,
            image_size=img_size,
            exts=["jpg", "jpeg", "JPG", "png"],
            do_augment=False,
        )

        print("Encoding and Manipulating the aligned image...")
        cls_manipulation_amplitude = manipulation_amplitude
        interpreted_target_class = target_class
        if (
            target_class not in CelebAttrDataset.id_to_cls
            and f"No_{target_class}" in CelebAttrDataset.id_to_cls
        ):
            cls_manipulation_amplitude = -manipulation_amplitude
            interpreted_target_class = f"No_{target_class}"

        batch = data[0]["img"][None]

        semantic_latent = self.model.encode(batch.to(self.device))
        stochastic_latent = self.model.encode_stochastic(
            batch.to(self.device), semantic_latent, T=T_inv
        )

        cls_id = CelebAttrDataset.cls_to_id[interpreted_target_class]
        class_direction = self.classifier.classifier.weight[cls_id]
        normalized_class_direction = F.normalize(class_direction[None, :], dim=1)

        normalized_semantic_latent = self.classifier.normalize(semantic_latent)
        normalized_manipulation_amp = cls_manipulation_amplitude * math.sqrt(512)
        normalized_manipulated_semantic_latent = (
            normalized_semantic_latent
            + normalized_manipulation_amp * normalized_class_direction
        )

        manipulated_semantic_latent = self.classifier.denormalize(
            normalized_manipulated_semantic_latent
        )

        # Render Manipulated image
        manipulated_img = self.model.render(
            stochastic_latent, manipulated_semantic_latent, T=T_step
        )[0]
        original_img = data[0]["img"]

        model_output = []
        out_path = Path(tempfile.mkdtemp()) / "original_aligned.png"
        save_image(convert2rgb(original_img), str(out_path))
        model_output.append(ModelOutput(image=out_path))

        out_path = Path(tempfile.mkdtemp()) / "manipulated_img.png"
        save_image(convert2rgb(manipulated_img, adjust_scale=False), str(out_path))
        model_output.append(ModelOutput(image=out_path))
        return model_output


def convert2rgb(img, adjust_scale=True):
    convert_img = torch.tensor(img)
    if adjust_scale:
        convert_img = (convert_img + 1) / 2
    return convert_img.cpu()

# FILE: ./__init__.py


# FILE: ./data_resize_celeba.py

import argparse
import multiprocessing
import os
import shutil
from functools import partial
from io import BytesIO
from multiprocessing import Process, Queue
from os.path import exists, join
from pathlib import Path

import lmdb
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import LSUNClass
from torchvision.transforms import functional as trans_fn
from tqdm import tqdm


def resize_and_convert(img, size, resample, quality=100):
    if size is not None:
        img = trans_fn.resize(img, size, resample)
        img = trans_fn.center_crop(img, size)

    buffer = BytesIO()
    img.save(buffer, format="webp", quality=quality)
    val = buffer.getvalue()

    return val


def resize_multiple(img,
                    sizes=(128, 256, 512, 1024),
                    resample=Image.LANCZOS,
                    quality=100):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size, resample, quality))

    return imgs


def resize_worker(idx, img, sizes, resample):
    img = img.convert("RGB")
    out = resize_multiple(img, sizes=sizes, resample=resample)
    return idx, out


class ConvertDataset(Dataset):
    def __init__(self, data, size) -> None:
        self.data = data
        self.size = size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        bytes = resize_and_convert(img, self.size, Image.LANCZOS, quality=100)
        return bytes


class ImageFolder(Dataset):
    def __init__(self, folder, ext='jpg'):
        super().__init__()
        paths = sorted([p for p in Path(f'{folder}').glob(f'*.{ext}')])
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = os.path.join(self.paths[index])
        img = Image.open(path)
        return img


if __name__ == "__main__":
    from tqdm import tqdm

    out_path = 'datasets/celeba.lmdb'
    in_path = 'datasets/celeba'
    ext = 'jpg'
    size = None

    dataset = ImageFolder(in_path, ext)
    print('len:', len(dataset))
    dataset = ConvertDataset(dataset, size)
    loader = DataLoader(dataset,
                        batch_size=50,
                        num_workers=12,
                        collate_fn=lambda x: x,
                        shuffle=False)

    target = os.path.expanduser(out_path)
    if os.path.exists(target):
        shutil.rmtree(target)

    with lmdb.open(target, map_size=1024**4, readahead=False) as env:
        with tqdm(total=len(dataset)) as progress:
            i = 0
            for batch in loader:
                with env.begin(write=True) as txn:
                    for img in batch:
                        key = f"{size}-{str(i).zfill(7)}".encode("utf-8")
                        # print(key)
                        txn.put(key, img)
                        i += 1
                        progress.update()
                # if i == 1000:
                #     break
                # if total == len(imgset):
                #     break

        with env.begin(write=True) as txn:
            txn.put("length".encode("utf-8"), str(i).encode("utf-8"))

# FILE: ./diffusion/diffusion.py

from .base import *
from dataclasses import dataclass


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}")
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


@dataclass
class SpacedDiffusionBeatGansConfig(GaussianDiffusionBeatGansConfig):
    use_timesteps: Tuple[int] = None

    def make_sampler(self):
        return SpacedDiffusionBeatGans(self)


class SpacedDiffusionBeatGans(GaussianDiffusionBeatGans):
    """
    A diffusion process which can skip steps in a base diffusion process.

    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """
    def __init__(self, conf: SpacedDiffusionBeatGansConfig):
        self.conf = conf
        self.use_timesteps = set(conf.use_timesteps)
        # how the new t's mapped to the old t's
        self.timestep_map = []
        self.original_num_steps = len(conf.betas)

        base_diffusion = GaussianDiffusionBeatGans(conf)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                # getting the new betas of the new timesteps
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        conf.betas = np.array(new_betas)
        super().__init__(conf)

    def p_mean_variance(self, model: Model, *args, **kwargs):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args,
                                       **kwargs)

    def training_losses(self, model: Model, *args, **kwargs):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args,
                                       **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args,
                                      **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args,
                                       **kwargs)

    def _wrap_model(self, model: Model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(model, self.timestep_map, self.rescale_timesteps,
                             self.original_num_steps)

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:
    """
    converting the supplied t's to the old t's scales.
    """
    def __init__(self, model, timestep_map, rescale_timesteps,
                 original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def forward(self, x, t, t_cond=None, **kwargs):
        """
        Args:
            t: t's with differrent ranges (can be << T due to smaller eval T) need to be converted to the original t's
            t_cond: the same as t but can be of different values
        """
        map_tensor = th.tensor(self.timestep_map,
                               device=t.device,
                               dtype=t.dtype)

        def do(t):
            new_ts = map_tensor[t]
            if self.rescale_timesteps:
                new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
            return new_ts

        if t_cond is not None:
            # support t_cond
            t_cond = do(t_cond)

        return self.model(x=x, t=do(t), t_cond=t_cond, **kwargs)

    def __getattr__(self, name):
        # allow for calling the model's methods
        if hasattr(self.model, name):
            func = getattr(self.model, name)
            return func
        raise AttributeError(name)

# FILE: ./diffusion/resample.py

from abc import ABC, abstractmethod

import numpy as np
import torch as th
import torch.distributed as dist


def create_named_schedule_sampler(name, diffusion):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.

    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if name == "uniform":
        return UniformSampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """
    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size, ), p=p)
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, num_timesteps):
        self._weights = np.ones([num_timesteps])

    def weights(self):
        return self._weights

# FILE: ./diffusion/__init__.py

from typing import Union

from .diffusion import SpacedDiffusionBeatGans, SpacedDiffusionBeatGansConfig

Sampler = Union[SpacedDiffusionBeatGans]
SamplerConfig = Union[SpacedDiffusionBeatGansConfig]

# FILE: ./diffusion/base.py

"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

from model.unet_autoenc import AutoencReturn
from config_base import BaseConfig
import enum
import math

import numpy as np
import torch as th
from model import *
from model.nn import mean_flat
from typing import NamedTuple, Tuple
from choices import *
from torch.cuda.amp import autocast
import torch.nn.functional as F

from dataclasses import dataclass


@dataclass
class GaussianDiffusionBeatGansConfig(BaseConfig):
    gen_type: GenerativeType
    betas: Tuple[float]
    model_type: ModelType
    model_mean_type: ModelMeanType
    model_var_type: ModelVarType
    loss_type: LossType
    rescale_timesteps: bool
    fp16: bool
    train_pred_xstart_detach: bool = True

    def make_sampler(self):
        return GaussianDiffusionBeatGans(self)


class GaussianDiffusionBeatGans:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """
    def __init__(self, conf: GaussianDiffusionBeatGansConfig):
        self.conf = conf
        self.model_mean_type = conf.model_mean_type
        self.model_var_type = conf.model_var_type
        self.loss_type = conf.loss_type
        self.rescale_timesteps = conf.rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(conf.betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps, )

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod -
                                                   1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (betas * (1.0 - self.alphas_cumprod_prev) /
                                   (1.0 - self.alphas_cumprod))
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        self.posterior_mean_coef1 = (betas *
                                     np.sqrt(self.alphas_cumprod_prev) /
                                     (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) *
                                     np.sqrt(alphas) /
                                     (1.0 - self.alphas_cumprod))

    def training_losses(self,
                        model: Model,
                        x_start: th.Tensor,
                        t: th.Tensor,
                        model_kwargs=None,
                        noise: th.Tensor = None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)

        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {'x_t': x_t}

        if self.loss_type in [
                LossType.mse,
                LossType.l1,
        ]:
            with autocast(self.conf.fp16):
                # x_t is static wrt. to the diffusion process
                model_forward = model.forward(x=x_t.detach(),
                                              t=self._scale_timesteps(t),
                                              x_start=x_start.detach(),
                                              **model_kwargs)
            model_output = model_forward.pred

            _model_output = model_output
            if self.conf.train_pred_xstart_detach:
                _model_output = _model_output.detach()
            # get the pred xstart
            p_mean_var = self.p_mean_variance(
                model=DummyModel(pred=_model_output),
                # gradient goes through x_t
                x=x_t,
                t=t,
                clip_denoised=False)
            terms['pred_xstart'] = p_mean_var['pred_xstart']

            # model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            target_types = {
                ModelMeanType.eps: noise,
            }
            target = target_types[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape

            if self.loss_type == LossType.mse:
                if self.model_mean_type == ModelMeanType.eps:
                    # (n, c, h, w) => (n, )
                    terms["mse"] = mean_flat((target - model_output)**2)
                else:
                    raise NotImplementedError()
            elif self.loss_type == LossType.l1:
                # (n, c, h, w) => (n, )
                terms["mse"] = mean_flat((target - model_output).abs())
            else:
                raise NotImplementedError()

            if "vb" in terms:
                # if learning the variance also use the vlb loss
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def sample(self,
               model: Model,
               shape=None,
               noise=None,
               cond=None,
               x_start=None,
               clip_denoised=True,
               model_kwargs=None,
               progress=False):
        """
        Args:
            x_start: given for the autoencoder
        """
        if model_kwargs is None:
            model_kwargs = {}
            if self.conf.model_type.has_autoenc():
                model_kwargs['x_start'] = x_start
                model_kwargs['cond'] = cond

        if self.conf.gen_type == GenerativeType.ddpm:
            return self.p_sample_loop(model,
                                      shape=shape,
                                      noise=noise,
                                      clip_denoised=clip_denoised,
                                      model_kwargs=model_kwargs,
                                      progress=progress)
        elif self.conf.gen_type == GenerativeType.ddim:
            return self.ddim_sample_loop(model,
                                         shape=shape,
                                         noise=noise,
                                         clip_denoised=clip_denoised,
                                         model_kwargs=model_kwargs,
                                         progress=progress)
        else:
            raise NotImplementedError()

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) *
            x_start)
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t,
                                        x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod,
                                            t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) *
            x_start + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod,
                                           t, x_start.shape) * noise)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) *
            x_start +
            _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) *
            x_t)
        posterior_variance = _extract_into_tensor(self.posterior_variance, t,
                                                  x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] ==
                posterior_log_variance_clipped.shape[0] == x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self,
                        model: Model,
                        x,
                        t,
                        clip_denoised=True,
                        denoised_fn=None,
                        model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B, )
        with autocast(self.conf.fp16):
            model_forward = model.forward(x=x,
                                          t=self._scale_timesteps(t),
                                          **model_kwargs)
        model_output = model_forward.pred

        if self.model_var_type in [
                ModelVarType.fixed_large, ModelVarType.fixed_small
        ]:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.fixed_large: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(
                        np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.fixed_small: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t,
                                                      x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type in [
                ModelMeanType.eps,
        ]:
            if self.model_mean_type == ModelMeanType.eps:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t,
                                                  eps=model_output))
            else:
                raise NotImplementedError()
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t)
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (model_mean.shape == model_log_variance.shape ==
                pred_xstart.shape == x.shape)
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            'model_forward': model_forward,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t,
                                     x_t.shape) * x_t -
                _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t,
                                     x_t.shape) * eps)

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape)
            * xprev - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t)

    def _predict_xstart_from_scaled_xstart(self, t, scaled_xstart):
        return scaled_xstart * _extract_into_tensor(
            self.sqrt_recip_alphas_cumprod, t, scaled_xstart.shape)

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t,
                                     x_t.shape) * x_t -
                pred_xstart) / _extract_into_tensor(
                    self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _predict_eps_from_scaled_xstart(self, x_t, t, scaled_xstart):
        """
        Args:
            scaled_xstart: is supposed to be sqrt(alphacum) * x_0
        """
        # 1 / sqrt(1-alphabar) * (x_t - scaled xstart)
        return (x_t - scaled_xstart) / _extract_into_tensor(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            # scale t to be maxed out at 1000 steps
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (p_mean_var["mean"].float() +
                    p_mean_var["variance"] * gradient.float())
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs)

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t)
        return out

    def p_sample(
        self,
        model: Model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
                        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(cond_fn,
                                              out,
                                              x,
                                              t,
                                              model_kwargs=model_kwargs)
        sample = out["mean"] + nonzero_mask * th.exp(
            0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model: Model,
        shape=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model: Model,
        shape=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        if noise is not None:
            img = noise
        else:
            assert isinstance(shape, (tuple, list))
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            # t = th.tensor([i] * shape[0], device=device)
            t = th.tensor([i] * len(img), device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model: Model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn,
                                       out,
                                       x,
                                       t,
                                       model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t,
                                              x.shape)
        sigma = (eta * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) *
                 th.sqrt(1 - alpha_bar / alpha_bar_prev))
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (out["pred_xstart"] * th.sqrt(alpha_bar_prev) +
                     th.sqrt(1 - alpha_bar_prev - sigma**2) * eps)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
                        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model: Model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        NOTE: never used ? 
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape)
               * x - out["pred_xstart"]) / _extract_into_tensor(
                   self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t,
                                              x.shape)

        # Equation 12. reversed  (DDIM paper)  (th.sqrt == torch.sqrt)
        mean_pred = (out["pred_xstart"] * th.sqrt(alpha_bar_next) +
                     th.sqrt(1 - alpha_bar_next) * eps)

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample_loop(
        self,
        model: Model,
        x,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
        device=None,
    ):
        if device is None:
            device = next(model.parameters()).device
        sample_t = []
        xstart_t = []
        T = []
        indices = list(range(self.num_timesteps))
        sample = x
        for i in indices:
            t = th.tensor([i] * len(sample), device=device)
            with th.no_grad():
                out = self.ddim_reverse_sample(model,
                                               sample,
                                               t=t,
                                               clip_denoised=clip_denoised,
                                               denoised_fn=denoised_fn,
                                               model_kwargs=model_kwargs,
                                               eta=eta)
                sample = out['sample']
                # [1, ..., T]
                sample_t.append(sample)
                # [0, ...., T-1]
                xstart_t.append(out['pred_xstart'])
                # [0, ..., T-1] ready to use
                T.append(t)

        return {
            #  xT "
            'sample': sample,
            # (1, ..., T)
            'sample_t': sample_t,
            # xstart here is a bit different from sampling from T = T-1 to T = 0
            # may not be exact
            'xstart_t': xstart_t,
            'T': T,
        }

    def ddim_sample_loop(
        self,
        model: Model,
        shape=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model: Model,
        shape=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        if noise is not None:
            img = noise
        else:
            assert isinstance(shape, (tuple, list))
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:

            if isinstance(model_kwargs, list):
                # index dependent model kwargs
                # (T-1, ..., 0)
                _kwargs = model_kwargs[i]
            else:
                _kwargs = model_kwargs

            t = th.tensor([i] * len(img), device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=_kwargs,
                    eta=eta,
                )
                out['t'] = t
                yield out
                img = out["sample"]

    def _vb_terms_bpd(self,
                      model: Model,
                      x_start,
                      x_t,
                      t,
                      clip_denoised=True,
                      model_kwargs=None):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t)
        out = self.p_mean_variance(model,
                                   x_t,
                                   t,
                                   clip_denoised=clip_denoised,
                                   model_kwargs=model_kwargs)
        kl = normal_kl(true_mean, true_log_variance_clipped, out["mean"],
                       out["log_variance"])
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"])
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {
            "output": output,
            "pred_xstart": out["pred_xstart"],
            'model_forward': out['model_forward'],
        }

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size,
                      device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean,
                             logvar1=qt_log_variance,
                             mean2=0.0,
                             logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self,
                      model: Model,
                      x_start,
                      clip_denoised=True,
                      model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start)**2))
            eps = self._predict_eps_from_xstart(x_t, t_batch,
                                                out["pred_xstart"])
            mse.append(mean_flat((eps - noise)**2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start,
                           beta_end,
                           num_diffusion_timesteps,
                           dtype=np.float64)
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2)**2,
        )
    elif schedule_name == "const0.01":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.01] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.015":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.015] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.008":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.008] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.0065":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.0065] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.0055":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.0055] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.0045":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.0045] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.0035":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.0035] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.0025":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.0025] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.0015":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.0015] * num_diffusion_timesteps,
                        dtype=np.float64)
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (-1.0 + logvar2 - logvar1 + th.exp(logvar1 - logvar2) +
                  ((mean1 - mean2)**2) * th.exp(-logvar2))


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (
        1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min,
                 th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


class DummyModel(th.nn.Module):
    def __init__(self, pred):
        super().__init__()
        self.pred = pred

    def forward(self, *args, **kwargs):
        return DummyReturn(pred=self.pred)


class DummyReturn(NamedTuple):
    pred: th.Tensor
# FILE: ./config_base.py

import json
import os
from copy import deepcopy
from dataclasses import dataclass


@dataclass
class BaseConfig:
    def clone(self):
        return deepcopy(self)

    def inherit(self, another):
        """inherit common keys from a given config"""
        common_keys = set(self.__dict__.keys()) & set(another.__dict__.keys())
        for k in common_keys:
            setattr(self, k, getattr(another, k))

    def propagate(self):
        """push down the configuration to all members"""
        for k, v in self.__dict__.items():
            if isinstance(v, BaseConfig):
                v.inherit(self)
                v.propagate()

    def save(self, save_path):
        """save config to json file"""
        dirname = os.path.dirname(save_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        conf = self.as_dict_jsonable()
        with open(save_path, 'w') as f:
            json.dump(conf, f)

    def load(self, load_path):
        """load json config"""
        with open(load_path) as f:
            conf = json.load(f)
        self.from_dict(conf)

    def from_dict(self, dict, strict=False):
        for k, v in dict.items():
            if not hasattr(self, k):
                if strict:
                    raise ValueError(f"loading extra '{k}'")
                else:
                    print(f"loading extra '{k}'")
                    continue
            if isinstance(self.__dict__[k], BaseConfig):
                self.__dict__[k].from_dict(v)
            else:
                self.__dict__[k] = v

    def as_dict_jsonable(self):
        conf = {}
        for k, v in self.__dict__.items():
            if isinstance(v, BaseConfig):
                conf[k] = v.as_dict_jsonable()
            else:
                if jsonable(v):
                    conf[k] = v
                else:
                    # ignore not jsonable
                    pass
        return conf


def jsonable(x):
    try:
        json.dumps(x)
        return True
    except TypeError:
        return False

# FILE: ./run_horse128_ddim.py

from templates import *
from templates_latent import *

if __name__ == '__main__':
    gpus = [0, 1, 2, 3]
    conf = horse128_ddpm()
    train(conf, gpus=gpus)

    gpus = [0, 1, 2, 3]
    conf.eval_programs = ['fid10']
    train(conf, gpus=gpus, mode='eval')
# FILE: ./dataset.py

import os
from io import BytesIO
from pathlib import Path

import lmdb
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, LSUNClass
import torch
import pandas as pd

import torchvision.transforms.functional as Ftrans


class ImageDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts=['jpg'],
        do_augment: bool = True,
        do_transform: bool = True,
        do_normalize: bool = True,
        sort_names=False,
        has_subdir: bool = True,
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size

        # relative paths (make it shorter, saves memory and faster to sort)
        if has_subdir:
            self.paths = [
                p.relative_to(folder) for ext in exts
                for p in Path(f'{folder}').glob(f'**/*.{ext}')
            ]
        else:
            self.paths = [
                p.relative_to(folder) for ext in exts
                for p in Path(f'{folder}').glob(f'*.{ext}')
            ]
        if sort_names:
            self.paths = sorted(self.paths)

        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = os.path.join(self.folder, self.paths[index])
        img = Image.open(path)
        # if the image is 'rgba'!
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return {'img': img, 'index': index}


class SubsetDataset(Dataset):
    def __init__(self, dataset, size):
        assert len(dataset) >= size
        self.dataset = dataset
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        assert index < self.size
        return self.dataset[index]


class BaseLMDB(Dataset):
    def __init__(self, path, original_resolution, zfill: int = 5):
        self.original_resolution = original_resolution
        self.zfill = zfill
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(
                txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.original_resolution}-{str(index).zfill(self.zfill)}'.encode(
                'utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        return img


def make_transform(
    image_size,
    flip_prob=0.5,
    crop_d2c=False,
):
    if crop_d2c:
        transform = [
            d2c_crop(),
            transforms.Resize(image_size),
        ]
    else:
        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
    transform.append(transforms.RandomHorizontalFlip(p=flip_prob))
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(transform)
    return transform


class FFHQlmdb(Dataset):
    def __init__(self,
                 path=os.path.expanduser('datasets/ffhq256.lmdb'),
                 image_size=256,
                 original_resolution=256,
                 split=None,
                 as_tensor: bool = True,
                 do_augment: bool = True,
                 do_normalize: bool = True,
                 **kwargs):
        self.original_resolution = original_resolution
        self.data = BaseLMDB(path, original_resolution, zfill=5)
        self.length = len(self.data)

        if split is None:
            self.offset = 0
        elif split == 'train':
            # last 60k
            self.length = self.length - 10000
            self.offset = 10000
        elif split == 'test':
            # first 10k
            self.length = 10000
            self.offset = 0
        else:
            raise NotImplementedError()

        transform = [
            transforms.Resize(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if as_tensor:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert index < self.length
        index = index + self.offset
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        return {'img': img, 'index': index}


class Crop:
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return Ftrans.crop(img, self.x1, self.y1, self.x2 - self.x1,
                           self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2)


def d2c_crop():
    # from D2C paper for CelebA dataset.
    cx = 89
    cy = 121
    x1 = cy - 64
    x2 = cy + 64
    y1 = cx - 64
    y2 = cx + 64
    return Crop(x1, x2, y1, y2)


class CelebAlmdb(Dataset):
    """
    also supports for d2c crop.
    """
    def __init__(self,
                 path,
                 image_size,
                 original_resolution=128,
                 split=None,
                 as_tensor: bool = True,
                 do_augment: bool = True,
                 do_normalize: bool = True,
                 crop_d2c: bool = False,
                 **kwargs):
        self.original_resolution = original_resolution
        self.data = BaseLMDB(path, original_resolution, zfill=7)
        self.length = len(self.data)
        self.crop_d2c = crop_d2c

        if split is None:
            self.offset = 0
        else:
            raise NotImplementedError()

        if crop_d2c:
            transform = [
                d2c_crop(),
                transforms.Resize(image_size),
            ]
        else:
            transform = [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
            ]

        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if as_tensor:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert index < self.length
        index = index + self.offset
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        return {'img': img, 'index': index}


class Horse_lmdb(Dataset):
    def __init__(self,
                 path=os.path.expanduser('datasets/horse256.lmdb'),
                 image_size=128,
                 original_resolution=256,
                 do_augment: bool = True,
                 do_transform: bool = True,
                 do_normalize: bool = True,
                 **kwargs):
        self.original_resolution = original_resolution
        print(path)
        self.data = BaseLMDB(path, original_resolution, zfill=7)
        self.length = len(self.data)

        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        return {'img': img, 'index': index}


class Bedroom_lmdb(Dataset):
    def __init__(self,
                 path=os.path.expanduser('datasets/bedroom256.lmdb'),
                 image_size=128,
                 original_resolution=256,
                 do_augment: bool = True,
                 do_transform: bool = True,
                 do_normalize: bool = True,
                 **kwargs):
        self.original_resolution = original_resolution
        print(path)
        self.data = BaseLMDB(path, original_resolution, zfill=7)
        self.length = len(self.data)

        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img = self.data[index]
        img = self.transform(img)
        return {'img': img, 'index': index}


class CelebAttrDataset(Dataset):

    id_to_cls = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
        'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
        'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
        'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
        'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
        'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
        'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
        'Wearing_Necklace', 'Wearing_Necktie', 'Young'
    ]
    cls_to_id = {v: k for k, v in enumerate(id_to_cls)}

    def __init__(self,
                 folder,
                 image_size=64,
                 attr_path=os.path.expanduser(
                     'datasets/celeba_anno/list_attr_celeba.txt'),
                 ext='png',
                 only_cls_name: str = None,
                 only_cls_value: int = None,
                 do_augment: bool = False,
                 do_transform: bool = True,
                 do_normalize: bool = True,
                 d2c: bool = False):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.ext = ext

        # relative paths (make it shorter, saves memory and faster to sort)
        paths = [
            str(p.relative_to(folder))
            for p in Path(f'{folder}').glob(f'**/*.{ext}')
        ]
        paths = [str(each).split('.')[0] + '.jpg' for each in paths]

        if d2c:
            transform = [
                d2c_crop(),
                transforms.Resize(image_size),
            ]
        else:
            transform = [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
            ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

        with open(attr_path) as f:
            # discard the top line
            f.readline()
            self.df = pd.read_csv(f, delim_whitespace=True)
            self.df = self.df[self.df.index.isin(paths)]

        if only_cls_name is not None:
            self.df = self.df[self.df[only_cls_name] == only_cls_value]

    def pos_count(self, cls_name):
        return (self.df[cls_name] == 1).sum()

    def neg_count(self, cls_name):
        return (self.df[cls_name] == -1).sum()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        name = row.name.split('.')[0]
        name = f'{name}.{self.ext}'

        path = os.path.join(self.folder, name)
        img = Image.open(path)

        labels = [0] * len(self.id_to_cls)
        for k, v in row.items():
            labels[self.cls_to_id[k]] = int(v)

        if self.transform is not None:
            img = self.transform(img)

        return {'img': img, 'index': index, 'labels': torch.tensor(labels)}


class CelebD2CAttrDataset(CelebAttrDataset):
    """
    the dataset is used in the D2C paper. 
    it has a specific crop from the original CelebA.
    """
    def __init__(self,
                 folder,
                 image_size=64,
                 attr_path=os.path.expanduser(
                     'datasets/celeba_anno/list_attr_celeba.txt'),
                 ext='jpg',
                 only_cls_name: str = None,
                 only_cls_value: int = None,
                 do_augment: bool = False,
                 do_transform: bool = True,
                 do_normalize: bool = True,
                 d2c: bool = True):
        super().__init__(folder,
                         image_size,
                         attr_path,
                         ext=ext,
                         only_cls_name=only_cls_name,
                         only_cls_value=only_cls_value,
                         do_augment=do_augment,
                         do_transform=do_transform,
                         do_normalize=do_normalize,
                         d2c=d2c)


class CelebAttrFewshotDataset(Dataset):
    def __init__(
        self,
        cls_name,
        K,
        img_folder,
        img_size=64,
        ext='png',
        seed=0,
        only_cls_name: str = None,
        only_cls_value: int = None,
        all_neg: bool = False,
        do_augment: bool = False,
        do_transform: bool = True,
        do_normalize: bool = True,
        d2c: bool = False,
    ) -> None:
        self.cls_name = cls_name
        self.K = K
        self.img_folder = img_folder
        self.ext = ext

        if all_neg:
            path = f'data/celeba_fewshots/K{K}_allneg_{cls_name}_{seed}.csv'
        else:
            path = f'data/celeba_fewshots/K{K}_{cls_name}_{seed}.csv'
        self.df = pd.read_csv(path, index_col=0)
        if only_cls_name is not None:
            self.df = self.df[self.df[only_cls_name] == only_cls_value]

        if d2c:
            transform = [
                d2c_crop(),
                transforms.Resize(img_size),
            ]
        else:
            transform = [
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
            ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def pos_count(self, cls_name):
        return (self.df[cls_name] == 1).sum()

    def neg_count(self, cls_name):
        return (self.df[cls_name] == -1).sum()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        name = row.name.split('.')[0]
        name = f'{name}.{self.ext}'

        path = os.path.join(self.img_folder, name)
        img = Image.open(path)

        # (1, 1)
        label = torch.tensor(int(row[self.cls_name])).unsqueeze(-1)

        if self.transform is not None:
            img = self.transform(img)

        return {'img': img, 'index': index, 'labels': label}


class CelebD2CAttrFewshotDataset(CelebAttrFewshotDataset):
    def __init__(self,
                 cls_name,
                 K,
                 img_folder,
                 img_size=64,
                 ext='jpg',
                 seed=0,
                 only_cls_name: str = None,
                 only_cls_value: int = None,
                 all_neg: bool = False,
                 do_augment: bool = False,
                 do_transform: bool = True,
                 do_normalize: bool = True,
                 is_negative=False,
                 d2c: bool = True) -> None:
        super().__init__(cls_name,
                         K,
                         img_folder,
                         img_size,
                         ext=ext,
                         seed=seed,
                         only_cls_name=only_cls_name,
                         only_cls_value=only_cls_value,
                         all_neg=all_neg,
                         do_augment=do_augment,
                         do_transform=do_transform,
                         do_normalize=do_normalize,
                         d2c=d2c)
        self.is_negative = is_negative


class CelebHQAttrDataset(Dataset):
    id_to_cls = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
        'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
        'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
        'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
        'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
        'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
        'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
        'Wearing_Necklace', 'Wearing_Necktie', 'Young'
    ]
    cls_to_id = {v: k for k, v in enumerate(id_to_cls)}

    def __init__(self,
                 path=os.path.expanduser('datasets/celebahq256.lmdb'),
                 image_size=None,
                 attr_path=os.path.expanduser(
                     'datasets/celeba_anno/CelebAMask-HQ-attribute-anno.txt'),
                 original_resolution=256,
                 do_augment: bool = False,
                 do_transform: bool = True,
                 do_normalize: bool = True):
        super().__init__()
        self.image_size = image_size
        self.data = BaseLMDB(path, original_resolution, zfill=5)

        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

        with open(attr_path) as f:
            # discard the top line
            f.readline()
            self.df = pd.read_csv(f, delim_whitespace=True)

    def pos_count(self, cls_name):
        return (self.df[cls_name] == 1).sum()

    def neg_count(self, cls_name):
        return (self.df[cls_name] == -1).sum()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = row.name
        img_idx, ext = img_name.split('.')
        img = self.data[img_idx]

        labels = [0] * len(self.id_to_cls)
        for k, v in row.items():
            labels[self.cls_to_id[k]] = int(v)

        if self.transform is not None:
            img = self.transform(img)
        return {'img': img, 'index': index, 'labels': torch.tensor(labels)}


class CelebHQAttrFewshotDataset(Dataset):
    def __init__(self,
                 cls_name,
                 K,
                 path,
                 image_size,
                 original_resolution=256,
                 do_augment: bool = False,
                 do_transform: bool = True,
                 do_normalize: bool = True):
        super().__init__()
        self.image_size = image_size
        self.cls_name = cls_name
        self.K = K
        self.data = BaseLMDB(path, original_resolution, zfill=5)

        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

        self.df = pd.read_csv(f'data/celebahq_fewshots/K{K}_{cls_name}.csv',
                              index_col=0)

    def pos_count(self, cls_name):
        return (self.df[cls_name] == 1).sum()

    def neg_count(self, cls_name):
        return (self.df[cls_name] == -1).sum()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = row.name
        img_idx, ext = img_name.split('.')
        img = self.data[img_idx]

        # (1, 1)
        label = torch.tensor(int(row[self.cls_name])).unsqueeze(-1)

        if self.transform is not None:
            img = self.transform(img)

        return {'img': img, 'index': index, 'labels': label}


class Repeat(Dataset):
    def __init__(self, dataset, new_len) -> None:
        super().__init__()
        self.dataset = dataset
        self.original_len = len(dataset)
        self.new_len = new_len

    def __len__(self):
        return self.new_len

    def __getitem__(self, index):
        index = index % self.original_len
        return self.dataset[index]

# FILE: ./templates.py

from experiment import *


def ddpm():
    """
    base configuration for all DDIM-based models.
    """
    conf = TrainConfig()
    conf.batch_size = 32
    conf.beatgans_gen_type = GenerativeType.ddim
    conf.beta_scheduler = 'linear'
    conf.data_name = 'ffhq'
    conf.diffusion_type = 'beatgans'
    conf.eval_ema_every_samples = 200_000
    conf.eval_every_samples = 200_000
    conf.fp16 = True
    conf.lr = 1e-4
    conf.model_name = ModelName.beatgans_ddpm
    conf.net_attn = (16, )
    conf.net_beatgans_attn_head = 1
    conf.net_beatgans_embed_channels = 512
    conf.net_ch_mult = (1, 2, 4, 8)
    conf.net_ch = 64
    conf.sample_size = 32
    conf.T_eval = 20
    conf.T = 1000
    conf.make_model_conf()
    return conf


def autoenc_base():
    """
    base configuration for all Diff-AE models.
    """
    conf = TrainConfig()
    conf.batch_size = 32
    conf.beatgans_gen_type = GenerativeType.ddim
    conf.beta_scheduler = 'linear'
    conf.data_name = 'ffhq'
    conf.diffusion_type = 'beatgans'
    conf.eval_ema_every_samples = 200_000
    conf.eval_every_samples = 200_000
    conf.fp16 = True
    conf.lr = 1e-4
    conf.model_name = ModelName.beatgans_autoenc
    conf.net_attn = (16, )
    conf.net_beatgans_attn_head = 1
    conf.net_beatgans_embed_channels = 512
    conf.net_beatgans_resnet_two_cond = True
    conf.net_ch_mult = (1, 2, 4, 8)
    conf.net_ch = 64
    conf.net_enc_channel_mult = (1, 2, 4, 8, 8)
    conf.net_enc_pool = 'adaptivenonzero'
    conf.sample_size = 32
    conf.T_eval = 20
    conf.T = 1000
    conf.make_model_conf()
    return conf


def ffhq64_ddpm():
    conf = ddpm()
    conf.data_name = 'ffhqlmdb256'
    conf.warmup = 0
    conf.total_samples = 72_000_000
    conf.scale_up_gpus(4)
    return conf


def ffhq64_autoenc():
    conf = autoenc_base()
    conf.data_name = 'ffhqlmdb256'
    conf.warmup = 0
    conf.total_samples = 72_000_000
    conf.net_ch_mult = (1, 2, 4, 8)
    conf.net_enc_channel_mult = (1, 2, 4, 8, 8)
    conf.eval_every_samples = 1_000_000
    conf.eval_ema_every_samples = 1_000_000
    conf.scale_up_gpus(4)
    conf.make_model_conf()
    return conf


def celeba64d2c_ddpm():
    conf = ffhq128_ddpm()
    conf.data_name = 'celebalmdb'
    conf.eval_every_samples = 10_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.total_samples = 72_000_000
    conf.name = 'celeba64d2c_ddpm'
    return conf


def celeba64d2c_autoenc():
    conf = ffhq64_autoenc()
    conf.data_name = 'celebalmdb'
    conf.eval_every_samples = 10_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.total_samples = 72_000_000
    conf.name = 'celeba64d2c_autoenc'
    return conf


def ffhq128_ddpm():
    conf = ddpm()
    conf.data_name = 'ffhqlmdb256'
    conf.warmup = 0
    conf.total_samples = 48_000_000
    conf.img_size = 128
    conf.net_ch = 128
    # channels:
    # 3 => 128 * 1 => 128 * 1 => 128 * 2 => 128 * 3 => 128 * 4
    # sizes:
    # 128 => 128 => 64 => 32 => 16 => 8
    conf.net_ch_mult = (1, 1, 2, 3, 4)
    conf.eval_every_samples = 1_000_000
    conf.eval_ema_every_samples = 1_000_000
    conf.scale_up_gpus(4)
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.make_model_conf()
    return conf


def ffhq128_autoenc_base():
    conf = autoenc_base()
    conf.data_name = 'ffhqlmdb256'
    conf.scale_up_gpus(4)
    conf.img_size = 128
    conf.net_ch = 128
    # final resolution = 8x8
    conf.net_ch_mult = (1, 1, 2, 3, 4)
    # final resolution = 4x4
    conf.net_enc_channel_mult = (1, 1, 2, 3, 4, 4)
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.make_model_conf()
    return conf


def ffhq256_autoenc():
    conf = ffhq128_autoenc_base()
    conf.img_size = 256
    conf.net_ch = 128
    conf.net_ch_mult = (1, 1, 2, 2, 4, 4)
    conf.net_enc_channel_mult = (1, 1, 2, 2, 4, 4, 4)
    conf.eval_every_samples = 10_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.total_samples = 200_000_000
    conf.batch_size = 64
    conf.make_model_conf()
    conf.name = 'ffhq256_autoenc'
    return conf


def ffhq256_autoenc_eco():
    conf = ffhq128_autoenc_base()
    conf.img_size = 256
    conf.net_ch = 128
    conf.net_ch_mult = (1, 1, 2, 2, 4, 4)
    conf.net_enc_channel_mult = (1, 1, 2, 2, 4, 4, 4)
    conf.eval_every_samples = 10_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.total_samples = 200_000_000
    conf.batch_size = 64
    conf.make_model_conf()
    conf.name = 'ffhq256_autoenc_eco'
    return conf


def ffhq128_ddpm_72M():
    conf = ffhq128_ddpm()
    conf.total_samples = 72_000_000
    conf.name = 'ffhq128_ddpm_72M'
    return conf


def ffhq128_autoenc_72M():
    conf = ffhq128_autoenc_base()
    conf.total_samples = 72_000_000
    conf.name = 'ffhq128_autoenc_72M'
    return conf


def ffhq128_ddpm_130M():
    conf = ffhq128_ddpm()
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.name = 'ffhq128_ddpm_130M'
    return conf


def ffhq128_autoenc_130M():
    conf = ffhq128_autoenc_base()
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.name = 'ffhq128_autoenc_130M'
    return conf


def horse128_ddpm():
    conf = ffhq128_ddpm()
    conf.data_name = 'horse256'
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.name = 'horse128_ddpm'
    return conf


def horse128_autoenc():
    conf = ffhq128_autoenc_base()
    conf.data_name = 'horse256'
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.name = 'horse128_autoenc'
    return conf


def bedroom128_ddpm():
    conf = ffhq128_ddpm()
    conf.data_name = 'bedroom256'
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.total_samples = 120_000_000
    conf.name = 'bedroom128_ddpm'
    return conf


def bedroom128_autoenc():
    conf = ffhq128_autoenc_base()
    conf.data_name = 'bedroom256'
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.total_samples = 120_000_000
    conf.name = 'bedroom128_autoenc'
    return conf


def pretrain_celeba64d2c_72M():
    conf = celeba64d2c_autoenc()
    conf.pretrain = PretrainConfig(
        name='72M',
        path=f'checkpoints/{celeba64d2c_autoenc().name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{celeba64d2c_autoenc().name}/latent.pkl'
    return conf


def pretrain_ffhq128_autoenc72M():
    conf = ffhq128_autoenc_base()
    conf.postfix = ''
    conf.pretrain = PretrainConfig(
        name='72M',
        path=f'checkpoints/{ffhq128_autoenc_72M().name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{ffhq128_autoenc_72M().name}/latent.pkl'
    return conf


def pretrain_ffhq128_autoenc130M():
    conf = ffhq128_autoenc_base()
    conf.pretrain = PretrainConfig(
        name='130M',
        path=f'checkpoints/{ffhq128_autoenc_130M().name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{ffhq128_autoenc_130M().name}/latent.pkl'
    return conf


def pretrain_ffhq256_autoenc():
    conf = ffhq256_autoenc()
    conf.pretrain = PretrainConfig(
        name='90M',
        path=f'checkpoints/{ffhq256_autoenc().name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{ffhq256_autoenc().name}/latent.pkl'
    return conf


def pretrain_horse128():
    conf = horse128_autoenc()
    conf.pretrain = PretrainConfig(
        name='82M',
        path=f'checkpoints/{horse128_autoenc().name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{horse128_autoenc().name}/latent.pkl'
    return conf


def pretrain_bedroom128():
    conf = bedroom128_autoenc()
    conf.pretrain = PretrainConfig(
        name='120M',
        path=f'checkpoints/{bedroom128_autoenc().name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{bedroom128_autoenc().name}/latent.pkl'
    return conf

# FILE: ./run_ffhq128_cls.py

from templates_cls import *
from experiment_classifier import *

if __name__ == '__main__':
    # need to first train the diffae autoencoding model & infer the latents
    # this requires only a single GPU.
    gpus = [0]
    conf = ffhq128_autoenc_cls()
    train_cls(conf, gpus=gpus)

    # after this you can do the manipulation!

# FILE: ./experiment.py

import copy
import json
import os
import re

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from numpy.lib.function_base import flip
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import *
from torch import nn
from torch.cuda import amp
from torch.distributions import Categorical
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataset import ConcatDataset, TensorDataset
from torchvision.utils import make_grid, save_image

from config import *
from dataset import *
from dist_utils import *
from lmdb_writer import *
from metrics import *
from renderer import *


class LitModel(pl.LightningModule):
    def __init__(self, conf: TrainConfig):
        super().__init__()
        assert conf.train_mode != TrainMode.manipulate
        if conf.seed is not None:
            pl.seed_everything(conf.seed)

        self.save_hyperparameters(conf.as_dict_jsonable())

        self.conf = conf

        self.model = conf.make_model_conf().make_model()
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()

        model_size = 0
        for param in self.model.parameters():
            model_size += param.data.nelement()
        print('Model params: %.2f M' % (model_size / 1024 / 1024))

        self.sampler = conf.make_diffusion_conf().make_sampler()
        self.eval_sampler = conf.make_eval_diffusion_conf().make_sampler()

        # this is shared for both model and latent
        self.T_sampler = conf.make_T_sampler()

        if conf.train_mode.use_latent_net():
            self.latent_sampler = conf.make_latent_diffusion_conf(
            ).make_sampler()
            self.eval_latent_sampler = conf.make_latent_eval_diffusion_conf(
            ).make_sampler()
        else:
            self.latent_sampler = None
            self.eval_latent_sampler = None

        # initial variables for consistent sampling
        self.register_buffer(
            'x_T',
            torch.randn(conf.sample_size, 3, conf.img_size, conf.img_size))

        if conf.pretrain is not None:
            print(f'loading pretrain ... {conf.pretrain.name}')
            state = torch.load(conf.pretrain.path, map_location='cpu')
            print('step:', state['global_step'])
            self.load_state_dict(state['state_dict'], strict=False)

        if conf.latent_infer_path is not None:
            print('loading latent stats ...')
            state = torch.load(conf.latent_infer_path)
            self.conds = state['conds']
            self.register_buffer('conds_mean', state['conds_mean'][None, :])
            self.register_buffer('conds_std', state['conds_std'][None, :])
        else:
            self.conds_mean = None
            self.conds_std = None

    def normalize(self, cond):
        cond = (cond - self.conds_mean.to(self.device)) / self.conds_std.to(
            self.device)
        return cond

    def denormalize(self, cond):
        cond = (cond * self.conds_std.to(self.device)) + self.conds_mean.to(
            self.device)
        return cond

    def sample(self, N, device, T=None, T_latent=None):
        if T is None:
            sampler = self.eval_sampler
            latent_sampler = self.latent_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler()
            latent_sampler = self.conf._make_latent_diffusion_conf(T_latent).make_sampler()

        noise = torch.randn(N,
                            3,
                            self.conf.img_size,
                            self.conf.img_size,
                            device=device)
        pred_img = render_uncondition(
            self.conf,
            self.ema_model,
            noise,
            sampler=sampler,
            latent_sampler=latent_sampler,
            conds_mean=self.conds_mean,
            conds_std=self.conds_std,
        )
        pred_img = (pred_img + 1) / 2
        return pred_img

    def render(self, noise, cond=None, T=None):
        if T is None:
            sampler = self.eval_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler()

        if cond is not None:
            pred_img = render_condition(self.conf,
                                        self.ema_model,
                                        noise,
                                        sampler=sampler,
                                        cond=cond)
        else:
            pred_img = render_uncondition(self.conf,
                                          self.ema_model,
                                          noise,
                                          sampler=sampler,
                                          latent_sampler=None)
        pred_img = (pred_img + 1) / 2
        return pred_img

    def encode(self, x):
        # TODO:
        assert self.conf.model_type.has_autoenc()
        cond = self.ema_model.encoder.forward(x)
        return cond

    def encode_stochastic(self, x, cond, T=None):
        if T is None:
            sampler = self.eval_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler()
        out = sampler.ddim_reverse_sample_loop(self.ema_model,
                                               x,
                                               model_kwargs={'cond': cond})
        return out['sample']

    def forward(self, noise=None, x_start=None, ema_model: bool = False):
        with amp.autocast(False):
            if ema_model:
                model = self.ema_model
            else:
                model = self.model
            gen = self.eval_sampler.sample(model=model,
                                           noise=noise,
                                           x_start=x_start)
            return gen

    def setup(self, stage=None) -> None:
        """
        make datasets & seeding each worker separately
        """
        ##############################################
        # NEED TO SET THE SEED SEPARATELY HERE
        if self.conf.seed is not None:
            seed = self.conf.seed * get_world_size() + self.global_rank
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print('local seed:', seed)
        ##############################################

        self.train_data = self.conf.make_dataset()
        print('train data:', len(self.train_data))
        self.val_data = self.train_data
        print('val data:', len(self.val_data))

    def _train_dataloader(self, drop_last=True):
        """
        really make the dataloader
        """
        # make sure to use the fraction of batch size
        # the batch size is global!
        conf = self.conf.clone()
        conf.batch_size = self.batch_size

        dataloader = conf.make_loader(self.train_data,
                                      shuffle=True,
                                      drop_last=drop_last)
        return dataloader

    def train_dataloader(self):
        """
        return the dataloader, if diffusion mode => return image dataset
        if latent mode => return the inferred latent dataset
        """
        print('on train dataloader start ...')
        if self.conf.train_mode.require_dataset_infer():
            if self.conds is None:
                # usually we load self.conds from a file
                # so we do not need to do this again!
                self.conds = self.infer_whole_dataset()
                # need to use float32! unless the mean & std will be off!
                # (1, c)
                self.conds_mean.data = self.conds.float().mean(dim=0,
                                                               keepdim=True)
                self.conds_std.data = self.conds.float().std(dim=0,
                                                             keepdim=True)
            print('mean:', self.conds_mean.mean(), 'std:',
                  self.conds_std.mean())

            # return the dataset with pre-calculated conds
            conf = self.conf.clone()
            conf.batch_size = self.batch_size
            data = TensorDataset(self.conds)
            return conf.make_loader(data, shuffle=True)
        else:
            return self._train_dataloader()

    @property
    def batch_size(self):
        """
        local batch size for each worker
        """
        ws = get_world_size()
        assert self.conf.batch_size % ws == 0
        return self.conf.batch_size // ws

    @property
    def num_samples(self):
        """
        (global) batch size * iterations
        """
        # batch size here is global!
        # global_step already takes into account the accum batches
        return self.global_step * self.conf.batch_size_effective

    def is_last_accum(self, batch_idx):
        """
        is it the last gradient accumulation loop? 
        used with gradient_accum > 1 and to see if the optimizer will perform "step" in this iteration or not
        """
        return (batch_idx + 1) % self.conf.accum_batches == 0

    def infer_whole_dataset(self,
                            with_render=False,
                            T_render=None,
                            render_save_path=None):
        """
        predicting the latents given images using the encoder

        Args:
            both_flips: include both original and flipped images; no need, it's not an improvement
            with_render: whether to also render the images corresponding to that latent
            render_save_path: lmdb output for the rendered images
        """
        data = self.conf.make_dataset()
        if isinstance(data, CelebAlmdb) and data.crop_d2c:
            # special case where we need the d2c crop
            data.transform = make_transform(self.conf.img_size,
                                            flip_prob=0,
                                            crop_d2c=True)
        else:
            data.transform = make_transform(self.conf.img_size, flip_prob=0)

        # data = SubsetDataset(data, 21)

        loader = self.conf.make_loader(
            data,
            shuffle=False,
            drop_last=False,
            batch_size=self.conf.batch_size_eval,
            parallel=True,
        )
        model = self.ema_model
        model.eval()
        conds = []

        if with_render:
            sampler = self.conf._make_diffusion_conf(
                T=T_render or self.conf.T_eval).make_sampler()

            if self.global_rank == 0:
                writer = LMDBImageWriter(render_save_path,
                                         format='webp',
                                         quality=100)
            else:
                writer = nullcontext()
        else:
            writer = nullcontext()

        with writer:
            for batch in tqdm(loader, total=len(loader), desc='infer'):
                with torch.no_grad():
                    # (n, c)
                    # print('idx:', batch['index'])
                    cond = model.encoder(batch['img'].to(self.device))

                    # used for reordering to match the original dataset
                    idx = batch['index']
                    idx = self.all_gather(idx)
                    if idx.dim() == 2:
                        idx = idx.flatten(0, 1)
                    argsort = idx.argsort()

                    if with_render:
                        noise = torch.randn(len(cond),
                                            3,
                                            self.conf.img_size,
                                            self.conf.img_size,
                                            device=self.device)
                        render = sampler.sample(model, noise=noise, cond=cond)
                        render = (render + 1) / 2
                        # print('render:', render.shape)
                        # (k, n, c, h, w)
                        render = self.all_gather(render)
                        if render.dim() == 5:
                            # (k*n, c)
                            render = render.flatten(0, 1)

                        # print('global_rank:', self.global_rank)

                        if self.global_rank == 0:
                            writer.put_images(render[argsort])

                    # (k, n, c)
                    cond = self.all_gather(cond)

                    if cond.dim() == 3:
                        # (k*n, c)
                        cond = cond.flatten(0, 1)

                    conds.append(cond[argsort].cpu())
                # break
        model.train()
        # (N, c) cpu

        conds = torch.cat(conds).float()
        return conds

    def training_step(self, batch, batch_idx):
        """
        given an input, calculate the loss function
        no optimization at this stage.
        """
        with amp.autocast(False):
            # batch size here is local!
            # forward
            if self.conf.train_mode.require_dataset_infer():
                # this mode as pre-calculated cond
                cond = batch[0]
                if self.conf.latent_znormalize:
                    cond = (cond - self.conds_mean.to(
                        self.device)) / self.conds_std.to(self.device)
            else:
                imgs, idxs = batch['img'], batch['index']
                # print(f'(rank {self.global_rank}) batch size:', len(imgs))
                x_start = imgs

            if self.conf.train_mode == TrainMode.diffusion:
                """
                main training mode!!!
                """
                # with numpy seed we have the problem that the sample t's are related!
                t, weight = self.T_sampler.sample(len(x_start), x_start.device)
                losses = self.sampler.training_losses(model=self.model,
                                                      x_start=x_start,
                                                      t=t)
            elif self.conf.train_mode.is_latent_diffusion():
                """
                training the latent variables!
                """
                # diffusion on the latent
                t, weight = self.T_sampler.sample(len(cond), cond.device)
                latent_losses = self.latent_sampler.training_losses(
                    model=self.model.latent_net, x_start=cond, t=t)
                # train only do the latent diffusion
                losses = {
                    'latent': latent_losses['loss'],
                    'loss': latent_losses['loss']
                }
            else:
                raise NotImplementedError()

            loss = losses['loss'].mean()
            # divide by accum batches to make the accumulated gradient exact!
            for key in ['loss', 'vae', 'latent', 'mmd', 'chamfer', 'arg_cnt']:
                if key in losses:
                    losses[key] = self.all_gather(losses[key]).mean()

            if self.global_rank == 0:
                self.logger.experiment.add_scalar('loss', losses['loss'],
                                                  self.num_samples)
                for key in ['vae', 'latent', 'mmd', 'chamfer', 'arg_cnt']:
                    if key in losses:
                        self.logger.experiment.add_scalar(
                            f'loss/{key}', losses[key], self.num_samples)

        return {'loss': loss}

    def on_train_batch_end(self, outputs, batch, batch_idx: int,
                           dataloader_idx: int) -> None:
        """
        after each training step ...
        """
        if self.is_last_accum(batch_idx):
            # only apply ema on the last gradient accumulation step,
            # if it is the iteration that has optimizer.step()
            if self.conf.train_mode == TrainMode.latent_diffusion:
                # it trains only the latent hence change only the latent
                ema(self.model.latent_net, self.ema_model.latent_net,
                    self.conf.ema_decay)
            else:
                ema(self.model, self.ema_model, self.conf.ema_decay)

            # logging
            if self.conf.train_mode.require_dataset_infer():
                imgs = None
            else:
                imgs = batch['img']
            self.log_sample(x_start=imgs)
            self.evaluate_scores()

    def on_before_optimizer_step(self, optimizer: Optimizer,
                                 optimizer_idx: int) -> None:
        # fix the fp16 + clip grad norm problem with pytorch lightinng
        # this is the currently correct way to do it
        if self.conf.grad_clip > 0:
            # from trainer.params_grads import grads_norm, iter_opt_params
            params = [
                p for group in optimizer.param_groups for p in group['params']
            ]
            # print('before:', grads_norm(iter_opt_params(optimizer)))
            torch.nn.utils.clip_grad_norm_(params,
                                           max_norm=self.conf.grad_clip)
            # print('after:', grads_norm(iter_opt_params(optimizer)))

    def log_sample(self, x_start):
        """
        put images to the tensorboard
        """
        def do(model,
               postfix,
               use_xstart,
               save_real=False,
               no_latent_diff=False,
               interpolate=False):
            model.eval()
            with torch.no_grad():
                all_x_T = self.split_tensor(self.x_T)
                batch_size = min(len(all_x_T), self.conf.batch_size_eval)
                # allow for superlarge models
                loader = DataLoader(all_x_T, batch_size=batch_size)

                Gen = []
                for x_T in loader:
                    if use_xstart:
                        _xstart = x_start[:len(x_T)]
                    else:
                        _xstart = None

                    if self.conf.train_mode.is_latent_diffusion(
                    ) and not use_xstart:
                        # diffusion of the latent first
                        gen = render_uncondition(
                            conf=self.conf,
                            model=model,
                            x_T=x_T,
                            sampler=self.eval_sampler,
                            latent_sampler=self.eval_latent_sampler,
                            conds_mean=self.conds_mean,
                            conds_std=self.conds_std)
                    else:
                        if not use_xstart and self.conf.model_type.has_noise_to_cond(
                        ):
                            model: BeatGANsAutoencModel
                            # special case, it may not be stochastic, yet can sample
                            cond = torch.randn(len(x_T),
                                               self.conf.style_ch,
                                               device=self.device)
                            cond = model.noise_to_cond(cond)
                        else:
                            if interpolate:
                                with amp.autocast(self.conf.fp16):
                                    cond = model.encoder(_xstart)
                                    i = torch.randperm(len(cond))
                                    cond = (cond + cond[i]) / 2
                            else:
                                cond = None
                        gen = self.eval_sampler.sample(model=model,
                                                       noise=x_T,
                                                       cond=cond,
                                                       x_start=_xstart)
                    Gen.append(gen)

                gen = torch.cat(Gen)
                gen = self.all_gather(gen)
                if gen.dim() == 5:
                    # (n, c, h, w)
                    gen = gen.flatten(0, 1)

                if save_real and use_xstart:
                    # save the original images to the tensorboard
                    real = self.all_gather(_xstart)
                    if real.dim() == 5:
                        real = real.flatten(0, 1)

                    if self.global_rank == 0:
                        grid_real = (make_grid(real) + 1) / 2
                        self.logger.experiment.add_image(
                            f'sample{postfix}/real', grid_real,
                            self.num_samples)

                if self.global_rank == 0:
                    # save samples to the tensorboard
                    grid = (make_grid(gen) + 1) / 2
                    sample_dir = os.path.join(self.conf.logdir,
                                              f'sample{postfix}')
                    if not os.path.exists(sample_dir):
                        os.makedirs(sample_dir)
                    path = os.path.join(sample_dir,
                                        '%d.png' % self.num_samples)
                    save_image(grid, path)
                    self.logger.experiment.add_image(f'sample{postfix}', grid,
                                                     self.num_samples)
            model.train()

        if self.conf.sample_every_samples > 0 and is_time(
                self.num_samples, self.conf.sample_every_samples,
                self.conf.batch_size_effective):

            if self.conf.train_mode.require_dataset_infer():
                do(self.model, '', use_xstart=False)
                do(self.ema_model, '_ema', use_xstart=False)
            else:
                if self.conf.model_type.has_autoenc(
                ) and self.conf.model_type.can_sample():
                    do(self.model, '', use_xstart=False)
                    do(self.ema_model, '_ema', use_xstart=False)
                    # autoencoding mode
                    do(self.model, '_enc', use_xstart=True, save_real=True)
                    do(self.ema_model,
                       '_enc_ema',
                       use_xstart=True,
                       save_real=True)
                elif self.conf.train_mode.use_latent_net():
                    do(self.model, '', use_xstart=False)
                    do(self.ema_model, '_ema', use_xstart=False)
                    # autoencoding mode
                    do(self.model, '_enc', use_xstart=True, save_real=True)
                    do(self.model,
                       '_enc_nodiff',
                       use_xstart=True,
                       save_real=True,
                       no_latent_diff=True)
                    do(self.ema_model,
                       '_enc_ema',
                       use_xstart=True,
                       save_real=True)
                else:
                    do(self.model, '', use_xstart=True, save_real=True)
                    do(self.ema_model, '_ema', use_xstart=True, save_real=True)

    def evaluate_scores(self):
        """
        evaluate FID and other scores during training (put to the tensorboard)
        For, FID. It is a fast version with 5k images (gold standard is 50k).
        Don't use its results in the paper!
        """
        def fid(model, postfix):
            score = evaluate_fid(self.eval_sampler,
                                 model,
                                 self.conf,
                                 device=self.device,
                                 train_data=self.train_data,
                                 val_data=self.val_data,
                                 latent_sampler=self.eval_latent_sampler,
                                 conds_mean=self.conds_mean,
                                 conds_std=self.conds_std)
            if self.global_rank == 0:
                self.logger.experiment.add_scalar(f'FID{postfix}', score,
                                                  self.num_samples)
                if not os.path.exists(self.conf.logdir):
                    os.makedirs(self.conf.logdir)
                with open(os.path.join(self.conf.logdir, 'eval.txt'),
                          'a') as f:
                    metrics = {
                        f'FID{postfix}': score,
                        'num_samples': self.num_samples,
                    }
                    f.write(json.dumps(metrics) + "\n")

        def lpips(model, postfix):
            if self.conf.model_type.has_autoenc(
            ) and self.conf.train_mode.is_autoenc():
                # {'lpips', 'ssim', 'mse'}
                score = evaluate_lpips(self.eval_sampler,
                                       model,
                                       self.conf,
                                       device=self.device,
                                       val_data=self.val_data,
                                       latent_sampler=self.eval_latent_sampler)

                if self.global_rank == 0:
                    for key, val in score.items():
                        self.logger.experiment.add_scalar(
                            f'{key}{postfix}', val, self.num_samples)

        if self.conf.eval_every_samples > 0 and self.num_samples > 0 and is_time(
                self.num_samples, self.conf.eval_every_samples,
                self.conf.batch_size_effective):
            print(f'eval fid @ {self.num_samples}')
            lpips(self.model, '')
            fid(self.model, '')

        if self.conf.eval_ema_every_samples > 0 and self.num_samples > 0 and is_time(
                self.num_samples, self.conf.eval_ema_every_samples,
                self.conf.batch_size_effective):
            print(f'eval fid ema @ {self.num_samples}')
            fid(self.ema_model, '_ema')
            # it's too slow
            # lpips(self.ema_model, '_ema')

    def configure_optimizers(self):
        out = {}
        if self.conf.optimizer == OptimizerType.adam:
            optim = torch.optim.Adam(self.model.parameters(),
                                     lr=self.conf.lr,
                                     weight_decay=self.conf.weight_decay)
        elif self.conf.optimizer == OptimizerType.adamw:
            optim = torch.optim.AdamW(self.model.parameters(),
                                      lr=self.conf.lr,
                                      weight_decay=self.conf.weight_decay)
        else:
            raise NotImplementedError()
        out['optimizer'] = optim
        if self.conf.warmup > 0:
            sched = torch.optim.lr_scheduler.LambdaLR(optim,
                                                      lr_lambda=WarmupLR(
                                                          self.conf.warmup))
            out['lr_scheduler'] = {
                'scheduler': sched,
                'interval': 'step',
            }
        return out

    def split_tensor(self, x):
        """
        extract the tensor for a corresponding "worker" in the batch dimension

        Args:
            x: (n, c)

        Returns: x: (n_local, c)
        """
        n = len(x)
        rank = self.global_rank
        world_size = get_world_size()
        # print(f'rank: {rank}/{world_size}')
        per_rank = n // world_size
        return x[rank * per_rank:(rank + 1) * per_rank]

    def test_step(self, batch, *args, **kwargs):
        """
        for the "eval" mode. 
        We first select what to do according to the "conf.eval_programs". 
        test_step will only run for "one iteration" (it's a hack!).
        
        We just want the multi-gpu support. 
        """
        # make sure you seed each worker differently!
        self.setup()

        # it will run only one step!
        print('global step:', self.global_step)
        """
        "infer" = predict the latent variables using the encoder on the whole dataset
        """
        if 'infer' in self.conf.eval_programs:
            if 'infer' in self.conf.eval_programs:
                print('infer ...')
                conds = self.infer_whole_dataset().float()
                # NOTE: always use this path for the latent.pkl files
                save_path = f'checkpoints/{self.conf.name}/latent.pkl'
            else:
                raise NotImplementedError()

            if self.global_rank == 0:
                conds_mean = conds.mean(dim=0)
                conds_std = conds.std(dim=0)
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                torch.save(
                    {
                        'conds': conds,
                        'conds_mean': conds_mean,
                        'conds_std': conds_std,
                    }, save_path)
        """
        "infer+render" = predict the latent variables using the encoder on the whole dataset
        THIS ALSO GENERATE CORRESPONDING IMAGES
        """
        # infer + reconstruction quality of the input
        for each in self.conf.eval_programs:
            if each.startswith('infer+render'):
                m = re.match(r'infer\+render([0-9]+)', each)
                if m is not None:
                    T = int(m[1])
                    self.setup()
                    print(f'infer + reconstruction T{T} ...')
                    conds = self.infer_whole_dataset(
                        with_render=True,
                        T_render=T,
                        render_save_path=
                        f'latent_infer_render{T}/{self.conf.name}.lmdb',
                    )
                    save_path = f'latent_infer_render{T}/{self.conf.name}.pkl'
                    conds_mean = conds.mean(dim=0)
                    conds_std = conds.std(dim=0)
                    if not os.path.exists(os.path.dirname(save_path)):
                        os.makedirs(os.path.dirname(save_path))
                    torch.save(
                        {
                            'conds': conds,
                            'conds_mean': conds_mean,
                            'conds_std': conds_std,
                        }, save_path)

        # evals those "fidXX"
        """
        "fid<T>" = unconditional generation (conf.train_mode = diffusion).
            Note:   Diff. autoenc will still receive real images in this mode.
        "fid<T>,<T_latent>" = unconditional generation for latent models (conf.train_mode = latent_diffusion).
            Note:   Diff. autoenc will still NOT receive real images in this made.
                    but you need to make sure that the train_mode is latent_diffusion.
        """
        for each in self.conf.eval_programs:
            if each.startswith('fid'):
                m = re.match(r'fid\(([0-9]+),([0-9]+)\)', each)
                clip_latent_noise = False
                if m is not None:
                    # eval(T1,T2)
                    T = int(m[1])
                    T_latent = int(m[2])
                    print(f'evaluating FID T = {T}... latent T = {T_latent}')
                else:
                    m = re.match(r'fidclip\(([0-9]+),([0-9]+)\)', each)
                    if m is not None:
                        # fidclip(T1,T2)
                        T = int(m[1])
                        T_latent = int(m[2])
                        clip_latent_noise = True
                        print(
                            f'evaluating FID (clip latent noise) T = {T}... latent T = {T_latent}'
                        )
                    else:
                        # evalT
                        _, T = each.split('fid')
                        T = int(T)
                        T_latent = None
                        print(f'evaluating FID T = {T}...')

                self.train_dataloader()
                sampler = self.conf._make_diffusion_conf(T=T).make_sampler()
                if T_latent is not None:
                    latent_sampler = self.conf._make_latent_diffusion_conf(
                        T=T_latent).make_sampler()
                else:
                    latent_sampler = None

                conf = self.conf.clone()
                conf.eval_num_images = 50_000
                score = evaluate_fid(
                    sampler,
                    self.ema_model,
                    conf,
                    device=self.device,
                    train_data=self.train_data,
                    val_data=self.val_data,
                    latent_sampler=latent_sampler,
                    conds_mean=self.conds_mean,
                    conds_std=self.conds_std,
                    remove_cache=False,
                    clip_latent_noise=clip_latent_noise,
                )
                if T_latent is None:
                    self.log(f'fid_ema_T{T}', score)
                else:
                    name = 'fid'
                    if clip_latent_noise:
                        name += '_clip'
                    name += f'_ema_T{T}_Tlatent{T_latent}'
                    self.log(name, score)
        """
        "recon<T>" = reconstruction & autoencoding (without noise inversion)
        """
        for each in self.conf.eval_programs:
            if each.startswith('recon'):
                self.model: BeatGANsAutoencModel
                _, T = each.split('recon')
                T = int(T)
                print(f'evaluating reconstruction T = {T}...')

                sampler = self.conf._make_diffusion_conf(T=T).make_sampler()

                conf = self.conf.clone()
                # eval whole val dataset
                conf.eval_num_images = len(self.val_data)
                # {'lpips', 'mse', 'ssim'}
                score = evaluate_lpips(sampler,
                                       self.ema_model,
                                       conf,
                                       device=self.device,
                                       val_data=self.val_data,
                                       latent_sampler=None)
                for k, v in score.items():
                    self.log(f'{k}_ema_T{T}', v)
        """
        "inv<T>" = reconstruction with noise inversion
        """
        for each in self.conf.eval_programs:
            if each.startswith('inv'):
                self.model: BeatGANsAutoencModel
                _, T = each.split('inv')
                T = int(T)
                print(
                    f'evaluating reconstruction with noise inversion T = {T}...'
                )

                sampler = self.conf._make_diffusion_conf(T=T).make_sampler()

                conf = self.conf.clone()
                # eval whole val dataset
                conf.eval_num_images = len(self.val_data)
                # {'lpips', 'mse', 'ssim'}
                score = evaluate_lpips(sampler,
                                       self.ema_model,
                                       conf,
                                       device=self.device,
                                       val_data=self.val_data,
                                       latent_sampler=None,
                                       use_inverted_noise=True)
                for k, v in score.items():
                    self.log(f'{k}_inv_ema_T{T}', v)


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(target_dict[key].data * decay +
                                    source_dict[key].data * (1 - decay))


class WarmupLR:
    def __init__(self, warmup) -> None:
        self.warmup = warmup

    def __call__(self, step):
        return min(step, self.warmup) / self.warmup


def is_time(num_samples, every, step_size):
    closest = (num_samples // every) * every
    return num_samples - closest < step_size


def train(conf: TrainConfig, gpus, nodes=1, mode: str = 'train'):
    print('conf:', conf.name)
    # assert not (conf.fp16 and conf.grad_clip > 0
    #             ), 'pytorch lightning has bug with amp + gradient clipping'
    model = LitModel(conf)

    if not os.path.exists(conf.logdir):
        os.makedirs(conf.logdir)
    checkpoint = ModelCheckpoint(dirpath=f'{conf.logdir}',
                                 save_last=True,
                                 save_top_k=1,
                                 every_n_train_steps=conf.save_every_samples //
                                 conf.batch_size_effective)
    checkpoint_path = f'{conf.logdir}/last.ckpt'
    print('ckpt path:', checkpoint_path)
    if os.path.exists(checkpoint_path):
        resume = checkpoint_path
        print('resume!')
    else:
        if conf.continue_from is not None:
            # continue from a checkpoint
            resume = conf.continue_from.path
        else:
            resume = None

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=conf.logdir,
                                             name=None,
                                             version='')

    # from pytorch_lightning.

    plugins = []
    if len(gpus) == 1 and nodes == 1:
        accelerator = None
    else:
        accelerator = 'ddp'
        from pytorch_lightning.plugins import DDPPlugin

        # important for working with gradient checkpoint
        plugins.append(DDPPlugin(find_unused_parameters=False))

    trainer = pl.Trainer(
        max_steps=conf.total_samples // conf.batch_size_effective,
        resume_from_checkpoint=resume,
        gpus=gpus,
        num_nodes=nodes,
        accelerator=accelerator,
        precision=16 if conf.fp16 else 32,
        callbacks=[
            checkpoint,
            LearningRateMonitor(),
        ],
        # clip in the model instead
        # gradient_clip_val=conf.grad_clip,
        replace_sampler_ddp=True,
        logger=tb_logger,
        accumulate_grad_batches=conf.accum_batches,
        plugins=plugins,
    )

    if mode == 'train':
        trainer.fit(model)
    elif mode == 'eval':
        # load the latest checkpoint
        # perform lpips
        # dummy loader to allow calling "test_step"
        dummy = DataLoader(TensorDataset(torch.tensor([0.] * conf.batch_size)),
                           batch_size=conf.batch_size)
        eval_path = conf.eval_path or checkpoint_path
        # conf.eval_num_images = 50
        print('loading from:', eval_path)
        state = torch.load(eval_path, map_location='cpu')
        print('step:', state['global_step'])
        model.load_state_dict(state['state_dict'])
        # trainer.fit(model)
        out = trainer.test(model, dataloaders=dummy)
        # first (and only) loader
        out = out[0]
        print(out)

        if get_rank() == 0:
            # save to tensorboard
            for k, v in out.items():
                tb_logger.experiment.add_scalar(
                    k, v, state['global_step'] * conf.batch_size_effective)

            # # save to file
            # # make it a dict of list
            # for k, v in out.items():
            #     out[k] = [v]
            tgt = f'evals/{conf.name}.txt'
            dirname = os.path.dirname(tgt)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            with open(tgt, 'a') as f:
                f.write(json.dumps(out) + "\n")
            # pd.DataFrame(out).to_csv(tgt)
    else:
        raise NotImplementedError()

# FILE: ./manipulate.py

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
batch = data[0]['img'][None]
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

# FILE: ./run_ffhq256_cls.py

from templates_cls import *
from experiment_classifier import *

if __name__ == '__main__':
    # need to first train the diffae autoencoding model & infer the latents
    # this requires only a single GPU.
    gpus = [0]
    conf = ffhq256_autoenc_cls()
    train_cls(conf, gpus=gpus)

    # after this you can do the manipulation!

# FILE: ./run_celeba64.py

from templates import *
from templates_latent import *

if __name__ == '__main__':
    # train the autoenc moodel
    # this can be run on 2080Ti's.
    gpus = [0, 1, 2, 3]
    conf = celeba64d2c_autoenc()
    train(conf, gpus=gpus)

    # infer the latents for training the latent DPM
    # NOTE: not gpu heavy, but more gpus can be of use!
    gpus = [0, 1, 2, 3]
    conf.eval_programs = ['infer']
    train(conf, gpus=gpus, mode='eval')

    # train the latent DPM
    # NOTE: only need a single gpu
    gpus = [0]
    conf = celeba64d2c_autoenc_latent()
    train(conf, gpus=gpus)

    # unconditional sampling score
    # NOTE: a lot of gpus can speed up this process
    gpus = [0, 1, 2, 3]
    conf.eval_programs = ['fid(10,10)']
    train(conf, gpus=gpus, mode='eval')
# FILE: ./experiment_classifier.py

from config import *
from dataset import *
import pandas as pd
import json
import os
import copy

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import *
import torch


class ZipLoader:
    def __init__(self, loaders):
        self.loaders = loaders

    def __len__(self):
        return len(self.loaders[0])

    def __iter__(self):
        for each in zip(*self.loaders):
            yield each


class ClsModel(pl.LightningModule):
    def __init__(self, conf: TrainConfig):
        super().__init__()
        assert conf.train_mode.is_manipulate()
        if conf.seed is not None:
            pl.seed_everything(conf.seed)

        self.save_hyperparameters(conf.as_dict_jsonable())
        self.conf = conf

        # preparations
        if conf.train_mode == TrainMode.manipulate:
            # this is only important for training!
            # the latent is freshly inferred to make sure it matches the image
            # manipulating latents require the base model
            self.model = conf.make_model_conf().make_model()
            self.ema_model = copy.deepcopy(self.model)
            self.model.requires_grad_(False)
            self.ema_model.requires_grad_(False)
            self.ema_model.eval()

            if conf.pretrain is not None:
                print(f'loading pretrain ... {conf.pretrain.name}')
                state = torch.load(conf.pretrain.path, map_location='cpu')
                print('step:', state['global_step'])
                self.load_state_dict(state['state_dict'], strict=False)

            # load the latent stats
            if conf.manipulate_znormalize:
                print('loading latent stats ...')
                state = torch.load(conf.latent_infer_path)
                self.conds = state['conds']
                self.register_buffer('conds_mean',
                                     state['conds_mean'][None, :])
                self.register_buffer('conds_std', state['conds_std'][None, :])
            else:
                self.conds_mean = None
                self.conds_std = None

        if conf.manipulate_mode in [ManipulateMode.celebahq_all]:
            num_cls = len(CelebAttrDataset.id_to_cls)
        elif conf.manipulate_mode.is_single_class():
            num_cls = 1
        else:
            raise NotImplementedError()

        # classifier
        if conf.train_mode == TrainMode.manipulate:
            # latent manipulation requires only a linear classifier
            self.classifier = nn.Linear(conf.style_ch, num_cls)
        else:
            raise NotImplementedError()

        self.ema_classifier = copy.deepcopy(self.classifier)

    def state_dict(self, *args, **kwargs):
        # don't save the base model
        out = {}
        for k, v in super().state_dict(*args, **kwargs).items():
            if k.startswith('model.'):
                pass
            elif k.startswith('ema_model.'):
                pass
            else:
                out[k] = v
        return out

    def load_state_dict(self, state_dict, strict: bool = None):
        if self.conf.train_mode == TrainMode.manipulate:
            # change the default strict => False
            if strict is None:
                strict = False
        else:
            if strict is None:
                strict = True
        return super().load_state_dict(state_dict, strict=strict)

    def normalize(self, cond):
        cond = (cond - self.conds_mean.to(self.device)) / self.conds_std.to(
            self.device)
        return cond

    def denormalize(self, cond):
        cond = (cond * self.conds_std.to(self.device)) + self.conds_mean.to(
            self.device)
        return cond

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
            # positive-unlabeled classifier needs to keep the class ratio 1:1
            # we use two dataloaders, one for each class, to stabiliize the training
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
        ##############################################
        # NEED TO SET THE SEED SEPARATELY HERE
        if self.conf.seed is not None:
            seed = self.conf.seed * get_world_size() + self.global_rank
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print('local seed:', seed)
        ##############################################

        self.train_data = self.load_dataset()
        if self.conf.manipulate_mode.is_fewshot():
            # repeat the dataset to be larger (speed up the training)
            if isinstance(self.train_data, list):
                # fewshot-allneg has two datasets
                # we resize them to be of equal sizes
                a, b = self.train_data
                self.train_data = [
                    Repeat(a, max(len(a), len(b))),
                    Repeat(b, max(len(a), len(b))),
                ]
            else:
                self.train_data = Repeat(self.train_data, 100_000)

    def train_dataloader(self):
        # make sure to use the fraction of batch size
        # the batch size is global!
        conf = self.conf.clone()
        conf.batch_size = self.batch_size
        if isinstance(self.train_data, list):
            dataloader = []
            for each in self.train_data:
                dataloader.append(
                    conf.make_loader(each, shuffle=True, drop_last=True))
            dataloader = ZipLoader(dataloader)
        else:
            dataloader = conf.make_loader(self.train_data,
                                          shuffle=True,
                                          drop_last=True)
        return dataloader

    @property
    def batch_size(self):
        ws = get_world_size()
        assert self.conf.batch_size % ws == 0
        return self.conf.batch_size // ws

    def training_step(self, batch, batch_idx):
        self.ema_model: BeatGANsAutoencModel
        if isinstance(batch, tuple):
            a, b = batch
            imgs = torch.cat([a['img'], b['img']])
            labels = torch.cat([a['labels'], b['labels']])
        else:
            imgs = batch['img']
            # print(f'({self.global_rank}) imgs:', imgs.shape)
            labels = batch['labels']

        if self.conf.train_mode == TrainMode.manipulate:
            self.ema_model.eval()
            with torch.no_grad():
                # (n, c)
                cond = self.ema_model.encoder(imgs)

            if self.conf.manipulate_znormalize:
                cond = self.normalize(cond)

            # (n, cls)
            pred = self.classifier.forward(cond)
            pred_ema = self.ema_classifier.forward(cond)
        elif self.conf.train_mode == TrainMode.manipulate_img:
            # (n, cls)
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
            gt = torch.where(labels > 0,
                             torch.ones_like(labels).float(),
                             torch.zeros_like(labels).float())
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

    def on_train_batch_end(self, outputs, batch, batch_idx: int,
                           dataloader_idx: int) -> None:
        ema(self.classifier, self.ema_classifier, self.conf.ema_decay)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.classifier.parameters(),
                                 lr=self.conf.lr,
                                 weight_decay=self.conf.weight_decay)
        return optim


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
        # every_n_train_steps=conf.save_every_samples //
        # conf.batch_size_effective,
    )
    checkpoint_path = f'{conf.logdir}/last.ckpt'
    if os.path.exists(checkpoint_path):
        resume = checkpoint_path
    else:
        if conf.continue_from is not None:
            # continue from a checkpoint
            resume = conf.continue_from.path
        else:
            resume = None

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=conf.logdir,
                                             name=None,
                                             version='')

    # from pytorch_lightning.

    plugins = []
    if len(gpus) == 1:
        accelerator = None
    else:
        accelerator = 'ddp'
        from pytorch_lightning.plugins import DDPPlugin
        # important for working with gradient checkpoint
        plugins.append(DDPPlugin(find_unused_parameters=False))

    trainer = pl.Trainer(
        max_steps=conf.total_samples // conf.batch_size_effective,
        resume_from_checkpoint=resume,
        gpus=gpus,
        accelerator=accelerator,
        precision=16 if conf.fp16 else 32,
        callbacks=[
            checkpoint,
        ],
        replace_sampler_ddp=True,
        logger=tb_logger,
        accumulate_grad_batches=conf.accum_batches,
        plugins=plugins,
    )
    trainer.fit(model)

# FILE: ./data_resize_horse.py

import argparse
import multiprocessing
import os
import shutil
from functools import partial
from io import BytesIO
from multiprocessing import Process, Queue
from os.path import exists, join

import lmdb
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import LSUNClass
from torchvision.transforms import functional as trans_fn
from tqdm import tqdm


def resize_and_convert(img, size, resample, quality=100):
    img = trans_fn.resize(img, size, resample)
    img = trans_fn.center_crop(img, size)
    buffer = BytesIO()
    img.save(buffer, format="webp", quality=quality)
    val = buffer.getvalue()

    return val


def resize_multiple(img,
                    sizes=(128, 256, 512, 1024),
                    resample=Image.LANCZOS,
                    quality=100):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size, resample, quality))

    return imgs


def resize_worker(idx, img, sizes, resample):
    img = img.convert("RGB")
    out = resize_multiple(img, sizes=sizes, resample=resample)
    return idx, out


class ConvertDataset(Dataset):
    def __init__(self, data) -> None:
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, _ = self.data[index]
        bytes = resize_and_convert(img, 256, Image.LANCZOS, quality=90)
        return bytes


if __name__ == "__main__":
    """
    converting lsun' original lmdb to our lmdb, which is somehow more performant.
    """
    from tqdm import tqdm

    # path to the original lsun's lmdb
    src_path = 'datasets/horse_train_lmdb'
    out_path = 'datasets/horse256.lmdb'

    dataset = LSUNClass(root=os.path.expanduser(src_path))
    dataset = ConvertDataset(dataset)
    loader = DataLoader(dataset,
                        batch_size=50,
                        num_workers=16,
                        collate_fn=lambda x: x)

    target = os.path.expanduser(out_path)
    if os.path.exists(target):
        shutil.rmtree(target)

    with lmdb.open(target, map_size=1024**4, readahead=False) as env:
        with tqdm(total=len(dataset)) as progress:
            i = 0
            for batch in loader:
                with env.begin(write=True) as txn:
                    for img in batch:
                        key = f"{256}-{str(i).zfill(7)}".encode("utf-8")
                        # print(key)
                        txn.put(key, img)
                        i += 1
                        progress.update()
                # if i == 1000:
                #     break
                # if total == len(imgset):
                #     break

        with env.begin(write=True) as txn:
            txn.put("length".encode("utf-8"), str(i).encode("utf-8"))

# FILE: ./model/nn.py

"""
Various utilities for neural networks.
"""

from enum import Enum
import math
from typing import Optional

import torch as th
import torch.nn as nn
import torch.utils.checkpoint

import torch.nn.functional as F


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    # @th.jit.script
    def forward(self, x):
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(min(32, channels), channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(-math.log(max_period) *
                   th.arange(start=0, end=half, dtype=th.float32) /
                   half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat(
            [embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def torch_checkpoint(func, args, flag, preserve_rng_state=False):
    # torch's gradient checkpoint works with automatic mixed precision, given torch >= 1.8
    if flag:
        return torch.utils.checkpoint.checkpoint(
            func, *args, preserve_rng_state=preserve_rng_state)
    else:
        return func(*args)

# FILE: ./model/unet_autoenc.py

from enum import Enum

import torch
from torch import Tensor
from torch.nn.functional import silu

from .latentnet import *
from .unet import *
from choices import *


@dataclass
class BeatGANsAutoencConfig(BeatGANsUNetConfig):
    # number of style channels
    enc_out_channels: int = 512
    enc_attn_resolutions: Tuple[int] = None
    enc_pool: str = 'depthconv'
    enc_num_res_block: int = 2
    enc_channel_mult: Tuple[int] = None
    enc_grad_checkpoint: bool = False
    latent_net_conf: MLPSkipNetConfig = None

    def make_model(self):
        return BeatGANsAutoencModel(self)


class BeatGANsAutoencModel(BeatGANsUNetModel):
    def __init__(self, conf: BeatGANsAutoencConfig):
        super().__init__(conf)
        self.conf = conf

        # having only time, cond
        self.time_embed = TimeStyleSeperateEmbed(
            time_channels=conf.model_channels,
            time_out_channels=conf.embed_channels,
        )

        self.encoder = BeatGANsEncoderConfig(
            image_size=conf.image_size,
            in_channels=conf.in_channels,
            model_channels=conf.model_channels,
            out_hid_channels=conf.enc_out_channels,
            out_channels=conf.enc_out_channels,
            num_res_blocks=conf.enc_num_res_block,
            attention_resolutions=(conf.enc_attn_resolutions
                                   or conf.attention_resolutions),
            dropout=conf.dropout,
            channel_mult=conf.enc_channel_mult or conf.channel_mult,
            use_time_condition=False,
            conv_resample=conf.conv_resample,
            dims=conf.dims,
            use_checkpoint=conf.use_checkpoint or conf.enc_grad_checkpoint,
            num_heads=conf.num_heads,
            num_head_channels=conf.num_head_channels,
            resblock_updown=conf.resblock_updown,
            use_new_attention_order=conf.use_new_attention_order,
            pool=conf.enc_pool,
        ).make_model()

        if conf.latent_net_conf is not None:
            self.latent_net = conf.latent_net_conf.make_model()

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        assert self.conf.is_stochastic
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample_z(self, n: int, device):
        assert self.conf.is_stochastic
        return torch.randn(n, self.conf.enc_out_channels, device=device)

    def noise_to_cond(self, noise: Tensor):
        raise NotImplementedError()
        assert self.conf.noise_net_conf is not None
        return self.noise_net.forward(noise)

    def encode(self, x):
        cond = self.encoder.forward(x)
        return {'cond': cond}

    @property
    def stylespace_sizes(self):
        modules = list(self.input_blocks.modules()) + list(
            self.middle_block.modules()) + list(self.output_blocks.modules())
        sizes = []
        for module in modules:
            if isinstance(module, ResBlock):
                linear = module.cond_emb_layers[-1]
                sizes.append(linear.weight.shape[0])
        return sizes

    def encode_stylespace(self, x, return_vector: bool = True):
        """
        encode to style space
        """
        modules = list(self.input_blocks.modules()) + list(
            self.middle_block.modules()) + list(self.output_blocks.modules())
        # (n, c)
        cond = self.encoder.forward(x)
        S = []
        for module in modules:
            if isinstance(module, ResBlock):
                # (n, c')
                s = module.cond_emb_layers.forward(cond)
                S.append(s)

        if return_vector:
            # (n, sum_c)
            return torch.cat(S, dim=1)
        else:
            return S

    def forward(self,
                x,
                t,
                y=None,
                x_start=None,
                cond=None,
                style=None,
                noise=None,
                t_cond=None,
                **kwargs):
        """
        Apply the model to an input batch.

        Args:
            x_start: the original image to encode
            cond: output of the encoder
            noise: random noise (to predict the cond)
        """

        if t_cond is None:
            t_cond = t

        if noise is not None:
            # if the noise is given, we predict the cond from noise
            cond = self.noise_to_cond(noise)

        if cond is None:
            if x is not None:
                assert len(x) == len(x_start), f'{len(x)} != {len(x_start)}'

            tmp = self.encode(x_start)
            cond = tmp['cond']

        if t is not None:
            _t_emb = timestep_embedding(t, self.conf.model_channels)
            _t_cond_emb = timestep_embedding(t_cond, self.conf.model_channels)
        else:
            # this happens when training only autoenc
            _t_emb = None
            _t_cond_emb = None

        if self.conf.resnet_two_cond:
            res = self.time_embed.forward(
                time_emb=_t_emb,
                cond=cond,
                time_cond_emb=_t_cond_emb,
            )
        else:
            raise NotImplementedError()

        if self.conf.resnet_two_cond:
            # two cond: first = time emb, second = cond_emb
            emb = res.time_emb
            cond_emb = res.emb
        else:
            # one cond = combined of both time and cond
            emb = res.emb
            cond_emb = None

        # override the style if given
        style = style or res.style

        assert (y is not None) == (
            self.conf.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        if self.conf.num_classes is not None:
            raise NotImplementedError()
            # assert y.shape == (x.shape[0], )
            # emb = emb + self.label_emb(y)

        # where in the model to supply time conditions
        enc_time_emb = emb
        mid_time_emb = emb
        dec_time_emb = emb
        # where in the model to supply style conditions
        enc_cond_emb = cond_emb
        mid_cond_emb = cond_emb
        dec_cond_emb = cond_emb

        # hs = []
        hs = [[] for _ in range(len(self.conf.channel_mult))]

        if x is not None:
            h = x.type(self.dtype)

            # input blocks
            k = 0
            for i in range(len(self.input_num_blocks)):
                for j in range(self.input_num_blocks[i]):
                    h = self.input_blocks[k](h,
                                             emb=enc_time_emb,
                                             cond=enc_cond_emb)

                    # print(i, j, h.shape)
                    hs[i].append(h)
                    k += 1
            assert k == len(self.input_blocks)

            # middle blocks
            h = self.middle_block(h, emb=mid_time_emb, cond=mid_cond_emb)
        else:
            # no lateral connections
            # happens when training only the autonecoder
            h = None
            hs = [[] for _ in range(len(self.conf.channel_mult))]

        # output blocks
        k = 0
        for i in range(len(self.output_num_blocks)):
            for j in range(self.output_num_blocks[i]):
                # take the lateral connection from the same layer (in reserve)
                # until there is no more, use None
                try:
                    lateral = hs[-i - 1].pop()
                    # print(i, j, lateral.shape)
                except IndexError:
                    lateral = None
                    # print(i, j, lateral)

                h = self.output_blocks[k](h,
                                          emb=dec_time_emb,
                                          cond=dec_cond_emb,
                                          lateral=lateral)
                k += 1

        pred = self.out(h)
        return AutoencReturn(pred=pred, cond=cond)


class AutoencReturn(NamedTuple):
    pred: Tensor
    cond: Tensor = None


class EmbedReturn(NamedTuple):
    # style and time
    emb: Tensor = None
    # time only
    time_emb: Tensor = None
    # style only (but could depend on time)
    style: Tensor = None


class TimeStyleSeperateEmbed(nn.Module):
    # embed only style
    def __init__(self, time_channels, time_out_channels):
        super().__init__()
        self.time_embed = nn.Sequential(
            linear(time_channels, time_out_channels),
            nn.SiLU(),
            linear(time_out_channels, time_out_channels),
        )
        self.style = nn.Identity()

    def forward(self, time_emb=None, cond=None, **kwargs):
        if time_emb is None:
            # happens with autoenc training mode
            time_emb = None
        else:
            time_emb = self.time_embed(time_emb)
        style = self.style(cond)
        return EmbedReturn(emb=style, time_emb=time_emb, style=style)

# FILE: ./model/unet.py

import math
from dataclasses import dataclass
from numbers import Number
from typing import NamedTuple, Tuple, Union

import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
from choices import *
from config_base import BaseConfig
from .blocks import *

from .nn import (conv_nd, linear, normalization, timestep_embedding,
                 torch_checkpoint, zero_module)


@dataclass
class BeatGANsUNetConfig(BaseConfig):
    image_size: int = 64
    in_channels: int = 3
    # base channels, will be multiplied
    model_channels: int = 64
    # output of the unet
    # suggest: 3
    # you only need 6 if you also model the variance of the noise prediction (usually we use an analytical variance hence 3)
    out_channels: int = 3
    # how many repeating resblocks per resolution
    # the decoding side would have "one more" resblock
    # default: 2
    num_res_blocks: int = 2
    # you can also set the number of resblocks specifically for the input blocks
    # default: None = above
    num_input_res_blocks: int = None
    # number of time embed channels and style channels
    embed_channels: int = 512
    # at what resolutions you want to do self-attention of the feature maps
    # attentions generally improve performance
    # default: [16]
    # beatgans: [32, 16, 8]
    attention_resolutions: Tuple[int] = (16, )
    # number of time embed channels
    time_embed_channels: int = None
    # dropout applies to the resblocks (on feature maps)
    dropout: float = 0.1
    channel_mult: Tuple[int] = (1, 2, 4, 8)
    input_channel_mult: Tuple[int] = None
    conv_resample: bool = True
    # always 2 = 2d conv
    dims: int = 2
    # don't use this, legacy from BeatGANs
    num_classes: int = None
    use_checkpoint: bool = False
    # number of attention heads
    num_heads: int = 1
    # or specify the number of channels per attention head
    num_head_channels: int = -1
    # what's this?
    num_heads_upsample: int = -1
    # use resblock for upscale/downscale blocks (expensive)
    # default: True (BeatGANs)
    resblock_updown: bool = True
    # never tried
    use_new_attention_order: bool = False
    resnet_two_cond: bool = False
    resnet_cond_channels: int = None
    # init the decoding conv layers with zero weights, this speeds up training
    # default: True (BeattGANs)
    resnet_use_zero_module: bool = True
    # gradient checkpoint the attention operation
    attn_checkpoint: bool = False

    def make_model(self):
        return BeatGANsUNetModel(self)


class BeatGANsUNetModel(nn.Module):
    def __init__(self, conf: BeatGANsUNetConfig):
        super().__init__()
        self.conf = conf

        if conf.num_heads_upsample == -1:
            self.num_heads_upsample = conf.num_heads

        self.dtype = th.float32

        self.time_emb_channels = conf.time_embed_channels or conf.model_channels
        self.time_embed = nn.Sequential(
            linear(self.time_emb_channels, conf.embed_channels),
            nn.SiLU(),
            linear(conf.embed_channels, conf.embed_channels),
        )

        if conf.num_classes is not None:
            self.label_emb = nn.Embedding(conf.num_classes,
                                          conf.embed_channels)

        ch = input_ch = int(conf.channel_mult[0] * conf.model_channels)
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(conf.dims, conf.in_channels, ch, 3, padding=1))
        ])

        kwargs = dict(
            use_condition=True,
            two_cond=conf.resnet_two_cond,
            use_zero_module=conf.resnet_use_zero_module,
            # style channels for the resnet block
            cond_emb_channels=conf.resnet_cond_channels,
        )

        self._feature_size = ch

        # input_block_chans = [ch]
        input_block_chans = [[] for _ in range(len(conf.channel_mult))]
        input_block_chans[0].append(ch)

        # number of blocks at each resolution
        self.input_num_blocks = [0 for _ in range(len(conf.channel_mult))]
        self.input_num_blocks[0] = 1
        self.output_num_blocks = [0 for _ in range(len(conf.channel_mult))]

        ds = 1
        resolution = conf.image_size
        for level, mult in enumerate(conf.input_channel_mult
                                     or conf.channel_mult):
            for _ in range(conf.num_input_res_blocks or conf.num_res_blocks):
                layers = [
                    ResBlockConfig(
                        ch,
                        conf.embed_channels,
                        conf.dropout,
                        out_channels=int(mult * conf.model_channels),
                        dims=conf.dims,
                        use_checkpoint=conf.use_checkpoint,
                        **kwargs,
                    ).make_model()
                ]
                ch = int(mult * conf.model_channels)
                if resolution in conf.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=conf.use_checkpoint
                            or conf.attn_checkpoint,
                            num_heads=conf.num_heads,
                            num_head_channels=conf.num_head_channels,
                            use_new_attention_order=conf.
                            use_new_attention_order,
                        ))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                # input_block_chans.append(ch)
                input_block_chans[level].append(ch)
                self.input_num_blocks[level] += 1
                # print(input_block_chans)
            if level != len(conf.channel_mult) - 1:
                resolution //= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlockConfig(
                            ch,
                            conf.embed_channels,
                            conf.dropout,
                            out_channels=out_ch,
                            dims=conf.dims,
                            use_checkpoint=conf.use_checkpoint,
                            down=True,
                            **kwargs,
                        ).make_model() if conf.
                        resblock_updown else Downsample(ch,
                                                        conf.conv_resample,
                                                        dims=conf.dims,
                                                        out_channels=out_ch)))
                ch = out_ch
                # input_block_chans.append(ch)
                input_block_chans[level + 1].append(ch)
                self.input_num_blocks[level + 1] += 1
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlockConfig(
                ch,
                conf.embed_channels,
                conf.dropout,
                dims=conf.dims,
                use_checkpoint=conf.use_checkpoint,
                **kwargs,
            ).make_model(),
            AttentionBlock(
                ch,
                use_checkpoint=conf.use_checkpoint or conf.attn_checkpoint,
                num_heads=conf.num_heads,
                num_head_channels=conf.num_head_channels,
                use_new_attention_order=conf.use_new_attention_order,
            ),
            ResBlockConfig(
                ch,
                conf.embed_channels,
                conf.dropout,
                dims=conf.dims,
                use_checkpoint=conf.use_checkpoint,
                **kwargs,
            ).make_model(),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(conf.channel_mult))[::-1]:
            for i in range(conf.num_res_blocks + 1):
                # print(input_block_chans)
                # ich = input_block_chans.pop()
                try:
                    ich = input_block_chans[level].pop()
                except IndexError:
                    # this happens only when num_res_block > num_enc_res_block
                    # we will not have enough lateral (skip) connecions for all decoder blocks
                    ich = 0
                # print('pop:', ich)
                layers = [
                    ResBlockConfig(
                        # only direct channels when gated
                        channels=ch + ich,
                        emb_channels=conf.embed_channels,
                        dropout=conf.dropout,
                        out_channels=int(conf.model_channels * mult),
                        dims=conf.dims,
                        use_checkpoint=conf.use_checkpoint,
                        # lateral channels are described here when gated
                        has_lateral=True if ich > 0 else False,
                        lateral_channels=None,
                        **kwargs,
                    ).make_model()
                ]
                ch = int(conf.model_channels * mult)
                if resolution in conf.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=conf.use_checkpoint
                            or conf.attn_checkpoint,
                            num_heads=self.num_heads_upsample,
                            num_head_channels=conf.num_head_channels,
                            use_new_attention_order=conf.
                            use_new_attention_order,
                        ))
                if level and i == conf.num_res_blocks:
                    resolution *= 2
                    out_ch = ch
                    layers.append(
                        ResBlockConfig(
                            ch,
                            conf.embed_channels,
                            conf.dropout,
                            out_channels=out_ch,
                            dims=conf.dims,
                            use_checkpoint=conf.use_checkpoint,
                            up=True,
                            **kwargs,
                        ).make_model() if (
                            conf.resblock_updown
                        ) else Upsample(ch,
                                        conf.conv_resample,
                                        dims=conf.dims,
                                        out_channels=out_ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self.output_num_blocks[level] += 1
                self._feature_size += ch

        # print(input_block_chans)
        # print('inputs:', self.input_num_blocks)
        # print('outputs:', self.output_num_blocks)

        if conf.resnet_use_zero_module:
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                zero_module(
                    conv_nd(conf.dims,
                            input_ch,
                            conf.out_channels,
                            3,
                            padding=1)),
            )
        else:
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                conv_nd(conf.dims, input_ch, conf.out_channels, 3, padding=1),
            )

    def forward(self, x, t, y=None, **kwargs):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.conf.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        # hs = []
        hs = [[] for _ in range(len(self.conf.channel_mult))]
        emb = self.time_embed(timestep_embedding(t, self.time_emb_channels))

        if self.conf.num_classes is not None:
            raise NotImplementedError()
            # assert y.shape == (x.shape[0], )
            # emb = emb + self.label_emb(y)

        # new code supports input_num_blocks != output_num_blocks
        h = x.type(self.dtype)
        k = 0
        for i in range(len(self.input_num_blocks)):
            for j in range(self.input_num_blocks[i]):
                h = self.input_blocks[k](h, emb=emb)
                # print(i, j, h.shape)
                hs[i].append(h)
                k += 1
        assert k == len(self.input_blocks)

        h = self.middle_block(h, emb=emb)
        k = 0
        for i in range(len(self.output_num_blocks)):
            for j in range(self.output_num_blocks[i]):
                # take the lateral connection from the same layer (in reserve)
                # until there is no more, use None
                try:
                    lateral = hs[-i - 1].pop()
                    # print(i, j, lateral.shape)
                except IndexError:
                    lateral = None
                    # print(i, j, lateral)
                h = self.output_blocks[k](h, emb=emb, lateral=lateral)
                k += 1

        h = h.type(x.dtype)
        pred = self.out(h)
        return Return(pred=pred)


class Return(NamedTuple):
    pred: th.Tensor


@dataclass
class BeatGANsEncoderConfig(BaseConfig):
    image_size: int
    in_channels: int
    model_channels: int
    out_hid_channels: int
    out_channels: int
    num_res_blocks: int
    attention_resolutions: Tuple[int]
    dropout: float = 0
    channel_mult: Tuple[int] = (1, 2, 4, 8)
    use_time_condition: bool = True
    conv_resample: bool = True
    dims: int = 2
    use_checkpoint: bool = False
    num_heads: int = 1
    num_head_channels: int = -1
    resblock_updown: bool = False
    use_new_attention_order: bool = False
    pool: str = 'adaptivenonzero'

    def make_model(self):
        return BeatGANsEncoderModel(self)


class BeatGANsEncoderModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.

    For usage, see UNet.
    """
    def __init__(self, conf: BeatGANsEncoderConfig):
        super().__init__()
        self.conf = conf
        self.dtype = th.float32

        if conf.use_time_condition:
            time_embed_dim = conf.model_channels * 4
            self.time_embed = nn.Sequential(
                linear(conf.model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )
        else:
            time_embed_dim = None

        ch = int(conf.channel_mult[0] * conf.model_channels)
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(conf.dims, conf.in_channels, ch, 3, padding=1))
        ])
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        resolution = conf.image_size
        for level, mult in enumerate(conf.channel_mult):
            for _ in range(conf.num_res_blocks):
                layers = [
                    ResBlockConfig(
                        ch,
                        time_embed_dim,
                        conf.dropout,
                        out_channels=int(mult * conf.model_channels),
                        dims=conf.dims,
                        use_condition=conf.use_time_condition,
                        use_checkpoint=conf.use_checkpoint,
                    ).make_model()
                ]
                ch = int(mult * conf.model_channels)
                if resolution in conf.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=conf.use_checkpoint,
                            num_heads=conf.num_heads,
                            num_head_channels=conf.num_head_channels,
                            use_new_attention_order=conf.
                            use_new_attention_order,
                        ))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(conf.channel_mult) - 1:
                resolution //= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlockConfig(
                            ch,
                            time_embed_dim,
                            conf.dropout,
                            out_channels=out_ch,
                            dims=conf.dims,
                            use_condition=conf.use_time_condition,
                            use_checkpoint=conf.use_checkpoint,
                            down=True,
                        ).make_model() if (
                            conf.resblock_updown
                        ) else Downsample(ch,
                                          conf.conv_resample,
                                          dims=conf.dims,
                                          out_channels=out_ch)))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlockConfig(
                ch,
                time_embed_dim,
                conf.dropout,
                dims=conf.dims,
                use_condition=conf.use_time_condition,
                use_checkpoint=conf.use_checkpoint,
            ).make_model(),
            AttentionBlock(
                ch,
                use_checkpoint=conf.use_checkpoint,
                num_heads=conf.num_heads,
                num_head_channels=conf.num_head_channels,
                use_new_attention_order=conf.use_new_attention_order,
            ),
            ResBlockConfig(
                ch,
                time_embed_dim,
                conf.dropout,
                dims=conf.dims,
                use_condition=conf.use_time_condition,
                use_checkpoint=conf.use_checkpoint,
            ).make_model(),
        )
        self._feature_size += ch
        if conf.pool == "adaptivenonzero":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                conv_nd(conf.dims, ch, conf.out_channels, 1),
                nn.Flatten(),
            )
        else:
            raise NotImplementedError(f"Unexpected {conf.pool} pooling")

    def forward(self, x, t=None, return_2d_feature=False):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        if self.conf.use_time_condition:
            emb = self.time_embed(timestep_embedding(t, self.model_channels))
        else:
            emb = None

        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb=emb)
            if self.conf.pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h, emb=emb)
        if self.conf.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            h = th.cat(results, axis=-1)
        else:
            h = h.type(x.dtype)

        h_2d = h
        h = self.out(h)

        if return_2d_feature:
            return h, h_2d
        else:
            return h

    def forward_flatten(self, x):
        """
        transform the last 2d feature into a flatten vector
        """
        h = self.out(x)
        return h


class SuperResModel(BeatGANsUNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """
    def __init__(self, image_size, in_channels, *args, **kwargs):
        super().__init__(image_size, in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width),
                                  mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)

# FILE: ./model/__init__.py

from typing import Union
from .unet import BeatGANsUNetModel, BeatGANsUNetConfig
from .unet_autoenc import BeatGANsAutoencConfig, BeatGANsAutoencModel

Model = Union[BeatGANsUNetModel, BeatGANsAutoencModel]
ModelConfig = Union[BeatGANsUNetConfig, BeatGANsAutoencConfig]

# FILE: ./model/latentnet.py

import math
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple, Tuple

import torch
from choices import *
from config_base import BaseConfig
from torch import nn
from torch.nn import init

from .blocks import *
from .nn import timestep_embedding
from .unet import *


class LatentNetType(Enum):
    none = 'none'
    # injecting inputs into the hidden layers
    skip = 'skip'


class LatentNetReturn(NamedTuple):
    pred: torch.Tensor = None


@dataclass
class MLPSkipNetConfig(BaseConfig):
    """
    default MLP for the latent DPM in the paper!
    """
    num_channels: int
    skip_layers: Tuple[int]
    num_hid_channels: int
    num_layers: int
    num_time_emb_channels: int = 64
    activation: Activation = Activation.silu
    use_norm: bool = True
    condition_bias: float = 1
    dropout: float = 0
    last_act: Activation = Activation.none
    num_time_layers: int = 2
    time_last_act: bool = False

    def make_model(self):
        return MLPSkipNet(self)


class MLPSkipNet(nn.Module):
    """
    concat x to hidden layers

    default MLP for the latent DPM in the paper!
    """
    def __init__(self, conf: MLPSkipNetConfig):
        super().__init__()
        self.conf = conf

        layers = []
        for i in range(conf.num_time_layers):
            if i == 0:
                a = conf.num_time_emb_channels
                b = conf.num_channels
            else:
                a = conf.num_channels
                b = conf.num_channels
            layers.append(nn.Linear(a, b))
            if i < conf.num_time_layers - 1 or conf.time_last_act:
                layers.append(conf.activation.get_act())
        self.time_embed = nn.Sequential(*layers)

        self.layers = nn.ModuleList([])
        for i in range(conf.num_layers):
            if i == 0:
                act = conf.activation
                norm = conf.use_norm
                cond = True
                a, b = conf.num_channels, conf.num_hid_channels
                dropout = conf.dropout
            elif i == conf.num_layers - 1:
                act = Activation.none
                norm = False
                cond = False
                a, b = conf.num_hid_channels, conf.num_channels
                dropout = 0
            else:
                act = conf.activation
                norm = conf.use_norm
                cond = True
                a, b = conf.num_hid_channels, conf.num_hid_channels
                dropout = conf.dropout

            if i in conf.skip_layers:
                a += conf.num_channels

            self.layers.append(
                MLPLNAct(
                    a,
                    b,
                    norm=norm,
                    activation=act,
                    cond_channels=conf.num_channels,
                    use_cond=cond,
                    condition_bias=conf.condition_bias,
                    dropout=dropout,
                ))
        self.last_act = conf.last_act.get_act()

    def forward(self, x, t, **kwargs):
        t = timestep_embedding(t, self.conf.num_time_emb_channels)
        cond = self.time_embed(t)
        h = x
        for i in range(len(self.layers)):
            if i in self.conf.skip_layers:
                # injecting input into the hidden layers
                h = torch.cat([h, x], dim=1)
            h = self.layers[i].forward(x=h, cond=cond)
        h = self.last_act(h)
        return LatentNetReturn(h)


class MLPLNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: bool,
        use_cond: bool,
        activation: Activation,
        cond_channels: int,
        condition_bias: float = 0,
        dropout: float = 0,
    ):
        super().__init__()
        self.activation = activation
        self.condition_bias = condition_bias
        self.use_cond = use_cond

        self.linear = nn.Linear(in_channels, out_channels)
        self.act = activation.get_act()
        if self.use_cond:
            self.linear_emb = nn.Linear(cond_channels, out_channels)
            self.cond_layers = nn.Sequential(self.act, self.linear_emb)
        if norm:
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = nn.Identity()

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.activation == Activation.relu:
                    init.kaiming_normal_(module.weight,
                                         a=0,
                                         nonlinearity='relu')
                elif self.activation == Activation.lrelu:
                    init.kaiming_normal_(module.weight,
                                         a=0.2,
                                         nonlinearity='leaky_relu')
                elif self.activation == Activation.silu:
                    init.kaiming_normal_(module.weight,
                                         a=0,
                                         nonlinearity='relu')
                else:
                    # leave it as default
                    pass

    def forward(self, x, cond=None):
        x = self.linear(x)
        if self.use_cond:
            # (n, c) or (n, c * 2)
            cond = self.cond_layers(cond)
            cond = (cond, None)

            # scale shift first
            x = x * (self.condition_bias + cond[0])
            if cond[1] is not None:
                x = x + cond[1]
            # then norm
            x = self.norm(x)
        else:
            # no condition
            x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x
# FILE: ./model/blocks.py

import math
from abc import abstractmethod
from dataclasses import dataclass
from numbers import Number

import torch as th
import torch.nn.functional as F
from choices import *
from config_base import BaseConfig
from torch import nn

from .nn import (avg_pool_nd, conv_nd, linear, normalization,
                 timestep_embedding, torch_checkpoint, zero_module)


class ScaleAt(Enum):
    after_norm = 'afternorm'


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """
    @abstractmethod
    def forward(self, x, emb=None, cond=None, lateral=None):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """
    def forward(self, x, emb=None, cond=None, lateral=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb=emb, cond=cond, lateral=lateral)
            else:
                x = layer(x)
        return x


@dataclass
class ResBlockConfig(BaseConfig):
    channels: int
    emb_channels: int
    dropout: float
    out_channels: int = None
    # condition the resblock with time (and encoder's output)
    use_condition: bool = True
    # whether to use 3x3 conv for skip path when the channels aren't matched
    use_conv: bool = False
    # dimension of conv (always 2 = 2d)
    dims: int = 2
    # gradient checkpoint
    use_checkpoint: bool = False
    up: bool = False
    down: bool = False
    # whether to condition with both time & encoder's output
    two_cond: bool = False
    # number of encoders' output channels
    cond_emb_channels: int = None
    # suggest: False
    has_lateral: bool = False
    lateral_channels: int = None
    # whether to init the convolution with zero weights
    # this is default from BeatGANs and seems to help learning
    use_zero_module: bool = True

    def __post_init__(self):
        self.out_channels = self.out_channels or self.channels
        self.cond_emb_channels = self.cond_emb_channels or self.emb_channels

    def make_model(self):
        return ResBlock(self)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    total layers:
        in_layers
        - norm
        - act
        - conv
        out_layers
        - norm
        - (modulation)
        - act
        - conv
    """
    def __init__(self, conf: ResBlockConfig):
        super().__init__()
        self.conf = conf

        #############################
        # IN LAYERS
        #############################
        assert conf.lateral_channels is None
        layers = [
            normalization(conf.channels),
            nn.SiLU(),
            conv_nd(conf.dims, conf.channels, conf.out_channels, 3, padding=1)
        ]
        self.in_layers = nn.Sequential(*layers)

        self.updown = conf.up or conf.down

        if conf.up:
            self.h_upd = Upsample(conf.channels, False, conf.dims)
            self.x_upd = Upsample(conf.channels, False, conf.dims)
        elif conf.down:
            self.h_upd = Downsample(conf.channels, False, conf.dims)
            self.x_upd = Downsample(conf.channels, False, conf.dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        #############################
        # OUT LAYERS CONDITIONS
        #############################
        if conf.use_condition:
            # condition layers for the out_layers
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(conf.emb_channels, 2 * conf.out_channels),
            )

            if conf.two_cond:
                self.cond_emb_layers = nn.Sequential(
                    nn.SiLU(),
                    linear(conf.cond_emb_channels, conf.out_channels),
                )
            #############################
            # OUT LAYERS (ignored when there is no condition)
            #############################
            # original version
            conv = conv_nd(conf.dims,
                           conf.out_channels,
                           conf.out_channels,
                           3,
                           padding=1)
            if conf.use_zero_module:
                # zere out the weights
                # it seems to help training
                conv = zero_module(conv)

            # construct the layers
            # - norm
            # - (modulation)
            # - act
            # - dropout
            # - conv
            layers = []
            layers += [
                normalization(conf.out_channels),
                nn.SiLU(),
                nn.Dropout(p=conf.dropout),
                conv,
            ]
            self.out_layers = nn.Sequential(*layers)

        #############################
        # SKIP LAYERS
        #############################
        if conf.out_channels == conf.channels:
            # cannot be used with gatedconv, also gatedconv is alsways used as the first block
            self.skip_connection = nn.Identity()
        else:
            if conf.use_conv:
                kernel_size = 3
                padding = 1
            else:
                kernel_size = 1
                padding = 0

            self.skip_connection = conv_nd(conf.dims,
                                           conf.channels,
                                           conf.out_channels,
                                           kernel_size,
                                           padding=padding)

    def forward(self, x, emb=None, cond=None, lateral=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        Args:
            x: input
            lateral: lateral connection from the encoder
        """
        return torch_checkpoint(self._forward, (x, emb, cond, lateral),
                                self.conf.use_checkpoint)

    def _forward(
        self,
        x,
        emb=None,
        cond=None,
        lateral=None,
    ):
        """
        Args:
            lateral: required if "has_lateral" and non-gated, with gated, it can be supplied optionally    
        """
        if self.conf.has_lateral:
            # lateral may be supplied even if it doesn't require
            # the model will take the lateral only if "has_lateral"
            assert lateral is not None
            x = th.cat([x, lateral], dim=1)

        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        if self.conf.use_condition:
            # it's possible that the network may not receieve the time emb
            # this happens with autoenc and setting the time_at
            if emb is not None:
                emb_out = self.emb_layers(emb).type(h.dtype)
            else:
                emb_out = None

            if self.conf.two_cond:
                # it's possible that the network is two_cond
                # but it doesn't get the second condition
                # in which case, we ignore the second condition
                # and treat as if the network has one condition
                if cond is None:
                    cond_out = None
                else:
                    cond_out = self.cond_emb_layers(cond).type(h.dtype)

                if cond_out is not None:
                    while len(cond_out.shape) < len(h.shape):
                        cond_out = cond_out[..., None]
            else:
                cond_out = None

            # this is the new refactored code
            h = apply_conditions(
                h=h,
                emb=emb_out,
                cond=cond_out,
                layers=self.out_layers,
                scale_bias=1,
                in_channels=self.conf.out_channels,
                up_down_layer=None,
            )

        return self.skip_connection(x) + h


def apply_conditions(
    h,
    emb=None,
    cond=None,
    layers: nn.Sequential = None,
    scale_bias: float = 1,
    in_channels: int = 512,
    up_down_layer: nn.Module = None,
):
    """
    apply conditions on the feature maps

    Args:
        emb: time conditional (ready to scale + shift)
        cond: encoder's conditional (read to scale + shift)
    """
    two_cond = emb is not None and cond is not None

    if emb is not None:
        # adjusting shapes
        while len(emb.shape) < len(h.shape):
            emb = emb[..., None]

    if two_cond:
        # adjusting shapes
        while len(cond.shape) < len(h.shape):
            cond = cond[..., None]
        # time first
        scale_shifts = [emb, cond]
    else:
        # "cond" is not used with single cond mode
        scale_shifts = [emb]

    # support scale, shift or shift only
    for i, each in enumerate(scale_shifts):
        if each is None:
            # special case: the condition is not provided
            a = None
            b = None
        else:
            if each.shape[1] == in_channels * 2:
                a, b = th.chunk(each, 2, dim=1)
            else:
                a = each
                b = None
        scale_shifts[i] = (a, b)

    # condition scale bias could be a list
    if isinstance(scale_bias, Number):
        biases = [scale_bias] * len(scale_shifts)
    else:
        # a list
        biases = scale_bias

    # default, the scale & shift are applied after the group norm but BEFORE SiLU
    pre_layers, post_layers = layers[0], layers[1:]

    # spilt the post layer to be able to scale up or down before conv
    # post layers will contain only the conv
    mid_layers, post_layers = post_layers[:-2], post_layers[-2:]

    h = pre_layers(h)
    # scale and shift for each condition
    for i, (scale, shift) in enumerate(scale_shifts):
        # if scale is None, it indicates that the condition is not provided
        if scale is not None:
            h = h * (biases[i] + scale)
            if shift is not None:
                h = h + shift
    h = mid_layers(h)

    # upscale or downscale if any just before the last conv
    if up_down_layer is not None:
        h = up_down_layer(h)
    h = post_layers(h)
    return h


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims,
                                self.channels,
                                self.out_channels,
                                3,
                                padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2),
                              mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims,
                              self.channels,
                              self.out_channels,
                              3,
                              stride=stride,
                              padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return torch_checkpoint(self._forward, (x, ), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial**2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch,
                                                                       dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale,
            k * scale)  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight,
                      v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """
    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim**2 + 1) / embed_dim**0.5)
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]

# FILE: ./sample.py

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
    ax[i].axis('off')  # Remove axis for cleaner visualization

# ✅ Save the figure
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Generated samples saved at: {output_path}")
# FILE: ./run_ffhq128.py

from templates import *
from templates_latent import *

if __name__ == '__main__':
    # train the autoenc moodel
    # this requires V100s.
    gpus = [0, 1, 2, 3]
    conf = ffhq128_autoenc_130M()
    train(conf, gpus=gpus)

    # infer the latents for training the latent DPM
    # NOTE: not gpu heavy, but more gpus can be of use!
    gpus = [0, 1, 2, 3]
    conf.eval_programs = ['infer']
    train(conf, gpus=gpus, mode='eval')

    # train the latent DPM
    # NOTE: only need a single gpu
    gpus = [0]
    conf = ffhq128_autoenc_latent()
    train(conf, gpus=gpus)

    # unconditional sampling score
    # NOTE: a lot of gpus can speed up this process
    gpus = [0, 1, 2, 3]
    conf.eval_programs = ['fid(10,10)']
    train(conf, gpus=gpus, mode='eval')
# FILE: ./dataset_util.py

import shutil
import os
from dist_utils import *


def use_cached_dataset_path(source_path, cache_path):
    if get_rank() == 0:
        if not os.path.exists(cache_path):
            # shutil.rmtree(cache_path)
            print(f'copying the data: {source_path} to {cache_path}')
            shutil.copytree(source_path, cache_path)
    barrier()
    return cache_path
# FILE: ./run_bedroom128.py

from templates import *
from templates_latent import *

if __name__ == '__main__':
    # train the autoenc moodel
    # this requires V100s.
    gpus = [0, 1, 2, 3]
    conf = bedroom128_autoenc()
    train(conf, gpus=gpus)

    # infer the latents for training the latent DPM
    # NOTE: not gpu heavy, but more gpus can be of use!
    gpus = [0, 1, 2, 3]
    conf.eval_programs = ['infer']
    train(conf, gpus=gpus, mode='eval')

    # train the latent DPM
    # NOTE: only need a single gpu
    gpus = [0]
    conf = bedroom128_autoenc_latent()
    train(conf, gpus=gpus)

    # unconditional sampling score
    # NOTE: a lot of gpus can speed up this process
    gpus = [0, 1, 2, 3]
    conf.eval_programs = ['fid(10,10)']
    train(conf, gpus=gpus, mode='eval')
# FILE: ./templates_latent.py

from templates import *


def latent_diffusion_config(conf: TrainConfig):
    conf.batch_size = 128
    conf.train_mode = TrainMode.latent_diffusion
    conf.latent_gen_type = GenerativeType.ddim
    conf.latent_loss_type = LossType.mse
    conf.latent_model_mean_type = ModelMeanType.eps
    conf.latent_model_var_type = ModelVarType.fixed_large
    conf.latent_rescale_timesteps = False
    conf.latent_clip_sample = False
    conf.latent_T_eval = 20
    conf.latent_znormalize = True
    conf.total_samples = 96_000_000
    conf.sample_every_samples = 400_000
    conf.eval_every_samples = 20_000_000
    conf.eval_ema_every_samples = 20_000_000
    conf.save_every_samples = 2_000_000
    return conf


def latent_diffusion128_config(conf: TrainConfig):
    conf = latent_diffusion_config(conf)
    conf.batch_size_eval = 32
    return conf


def latent_mlp_2048_norm_10layers(conf: TrainConfig):
    conf.net_latent_net_type = LatentNetType.skip
    conf.net_latent_layers = 10
    conf.net_latent_skip_layers = list(range(1, conf.net_latent_layers))
    conf.net_latent_activation = Activation.silu
    conf.net_latent_num_hid_channels = 2048
    conf.net_latent_use_norm = True
    conf.net_latent_condition_bias = 1
    return conf


def latent_mlp_2048_norm_20layers(conf: TrainConfig):
    conf = latent_mlp_2048_norm_10layers(conf)
    conf.net_latent_layers = 20
    conf.net_latent_skip_layers = list(range(1, conf.net_latent_layers))
    return conf


def latent_256_batch_size(conf: TrainConfig):
    conf.batch_size = 256
    conf.eval_ema_every_samples = 100_000_000
    conf.eval_every_samples = 100_000_000
    conf.sample_every_samples = 1_000_000
    conf.save_every_samples = 2_000_000
    conf.total_samples = 301_000_000
    return conf


def latent_512_batch_size(conf: TrainConfig):
    conf.batch_size = 512
    conf.eval_ema_every_samples = 100_000_000
    conf.eval_every_samples = 100_000_000
    conf.sample_every_samples = 1_000_000
    conf.save_every_samples = 5_000_000
    conf.total_samples = 501_000_000
    return conf


def latent_2048_batch_size(conf: TrainConfig):
    conf.batch_size = 2048
    conf.eval_ema_every_samples = 200_000_000
    conf.eval_every_samples = 200_000_000
    conf.sample_every_samples = 4_000_000
    conf.save_every_samples = 20_000_000
    conf.total_samples = 1_501_000_000
    return conf


def adamw_weight_decay(conf: TrainConfig):
    conf.optimizer = OptimizerType.adamw
    conf.weight_decay = 0.01
    return conf


def ffhq128_autoenc_latent():
    conf = pretrain_ffhq128_autoenc130M()
    conf = latent_diffusion128_config(conf)
    conf = latent_mlp_2048_norm_10layers(conf)
    conf = latent_256_batch_size(conf)
    conf = adamw_weight_decay(conf)
    conf.total_samples = 101_000_000
    conf.latent_loss_type = LossType.l1
    conf.latent_beta_scheduler = 'const0.008'
    conf.name = 'ffhq128_autoenc_latent'
    return conf


def ffhq256_autoenc_latent():
    conf = pretrain_ffhq256_autoenc()
    conf = latent_diffusion128_config(conf)
    conf = latent_mlp_2048_norm_10layers(conf)
    conf = latent_256_batch_size(conf)
    conf = adamw_weight_decay(conf)
    conf.total_samples = 101_000_000
    conf.latent_loss_type = LossType.l1
    conf.latent_beta_scheduler = 'const0.008'
    conf.eval_ema_every_samples = 200_000_000
    conf.eval_every_samples = 200_000_000
    conf.sample_every_samples = 4_000_000
    conf.name = 'ffhq256_autoenc_latent'
    return conf


def horse128_autoenc_latent():
    conf = pretrain_horse128()
    conf = latent_diffusion128_config(conf)
    conf = latent_2048_batch_size(conf)
    conf = latent_mlp_2048_norm_20layers(conf)
    conf.total_samples = 2_001_000_000
    conf.latent_beta_scheduler = 'const0.008'
    conf.latent_loss_type = LossType.l1
    conf.name = 'horse128_autoenc_latent'
    return conf


def bedroom128_autoenc_latent():
    conf = pretrain_bedroom128()
    conf = latent_diffusion128_config(conf)
    conf = latent_2048_batch_size(conf)
    conf = latent_mlp_2048_norm_20layers(conf)
    conf.total_samples = 2_001_000_000
    conf.latent_beta_scheduler = 'const0.008'
    conf.latent_loss_type = LossType.l1
    conf.name = 'bedroom128_autoenc_latent'
    return conf


def celeba64d2c_autoenc_latent():
    conf = pretrain_celeba64d2c_72M()
    conf = latent_diffusion_config(conf)
    conf = latent_512_batch_size(conf)
    conf = latent_mlp_2048_norm_10layers(conf)
    conf = adamw_weight_decay(conf)
    # just for the name
    conf.continue_from = PretrainConfig('200M',
                                        f'log-latent/{conf.name}/last.ckpt')
    conf.postfix = '_300M'
    conf.total_samples = 301_000_000
    conf.latent_beta_scheduler = 'const0.008'
    conf.latent_loss_type = LossType.l1
    conf.name = 'celeba64d2c_autoenc_latent'
    return conf

# FILE: ./choices.py

from enum import Enum
from torch import nn


class TrainMode(Enum):
    # manipulate mode = training the classifier
    manipulate = 'manipulate'
    # default trainin mode!
    diffusion = 'diffusion'
    # default latent training mode!
    # fitting the a DDPM to a given latent
    latent_diffusion = 'latentdiffusion'

    def is_manipulate(self):
        return self in [
            TrainMode.manipulate,
        ]

    def is_diffusion(self):
        return self in [
            TrainMode.diffusion,
            TrainMode.latent_diffusion,
        ]

    def is_autoenc(self):
        # the network possibly does autoencoding
        return self in [
            TrainMode.diffusion,
        ]

    def is_latent_diffusion(self):
        return self in [
            TrainMode.latent_diffusion,
        ]

    def use_latent_net(self):
        return self.is_latent_diffusion()

    def require_dataset_infer(self):
        """
        whether training in this mode requires the latent variables to be available?
        """
        # this will precalculate all the latents before hand
        # and the dataset will be all the predicted latents
        return self in [
            TrainMode.latent_diffusion,
            TrainMode.manipulate,
        ]


class ManipulateMode(Enum):
    """
    how to train the classifier to manipulate
    """
    # train on whole celeba attr dataset
    celebahq_all = 'celebahq_all'
    # celeba with D2C's crop
    d2c_fewshot = 'd2cfewshot'
    d2c_fewshot_allneg = 'd2cfewshotallneg'

    def is_celeba_attr(self):
        return self in [
            ManipulateMode.d2c_fewshot,
            ManipulateMode.d2c_fewshot_allneg,
            ManipulateMode.celebahq_all,
        ]

    def is_single_class(self):
        return self in [
            ManipulateMode.d2c_fewshot,
            ManipulateMode.d2c_fewshot_allneg,
        ]

    def is_fewshot(self):
        return self in [
            ManipulateMode.d2c_fewshot,
            ManipulateMode.d2c_fewshot_allneg,
        ]

    def is_fewshot_allneg(self):
        return self in [
            ManipulateMode.d2c_fewshot_allneg,
        ]


class ModelType(Enum):
    """
    Kinds of the backbone models
    """

    # unconditional ddpm
    ddpm = 'ddpm'
    # autoencoding ddpm cannot do unconditional generation
    autoencoder = 'autoencoder'

    def has_autoenc(self):
        return self in [
            ModelType.autoencoder,
        ]

    def can_sample(self):
        return self in [ModelType.ddpm]


class ModelName(Enum):
    """
    List of all supported model classes
    """

    beatgans_ddpm = 'beatgans_ddpm'
    beatgans_autoenc = 'beatgans_autoenc'


class ModelMeanType(Enum):
    """
    Which type of output the model predicts.
    """

    eps = 'eps'  # the model predicts epsilon


class ModelVarType(Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    # posterior beta_t
    fixed_small = 'fixed_small'
    # beta_t
    fixed_large = 'fixed_large'


class LossType(Enum):
    mse = 'mse'  # use raw MSE loss (and KL when learning variances)
    l1 = 'l1'


class GenerativeType(Enum):
    """
    How's a sample generated
    """

    ddpm = 'ddpm'
    ddim = 'ddim'


class OptimizerType(Enum):
    adam = 'adam'
    adamw = 'adamw'


class Activation(Enum):
    none = 'none'
    relu = 'relu'
    lrelu = 'lrelu'
    silu = 'silu'
    tanh = 'tanh'

    def get_act(self):
        if self == Activation.none:
            return nn.Identity()
        elif self == Activation.relu:
            return nn.ReLU()
        elif self == Activation.lrelu:
            return nn.LeakyReLU(negative_slope=0.2)
        elif self == Activation.silu:
            return nn.SiLU()
        elif self == Activation.tanh:
            return nn.Tanh()
        else:
            raise NotImplementedError()


class ManipulateLossType(Enum):
    bce = 'bce'
    mse = 'mse'
# FILE: ./lmdb_writer.py

from io import BytesIO

import lmdb
from PIL import Image

import torch

from contextlib import contextmanager
from torch.utils.data import Dataset
from multiprocessing import Process, Queue
import os
import shutil


def convert(x, format, quality=100):
    # to prevent locking!
    torch.set_num_threads(1)

    buffer = BytesIO()
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    x = x.to(torch.uint8)
    x = x.numpy()
    img = Image.fromarray(x)
    img.save(buffer, format=format, quality=quality)
    val = buffer.getvalue()
    return val


@contextmanager
def nullcontext():
    yield


class _WriterWroker(Process):
    def __init__(self, path, format, quality, zfill, q):
        super().__init__()
        if os.path.exists(path):
            shutil.rmtree(path)

        self.path = path
        self.format = format
        self.quality = quality
        self.zfill = zfill
        self.q = q
        self.i = 0

    def run(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        with lmdb.open(self.path, map_size=1024**4, readahead=False) as env:
            while True:
                job = self.q.get()
                if job is None:
                    break
                with env.begin(write=True) as txn:
                    for x in job:
                        key = f"{str(self.i).zfill(self.zfill)}".encode(
                            "utf-8")
                        x = convert(x, self.format, self.quality)
                        txn.put(key, x)
                        self.i += 1

            with env.begin(write=True) as txn:
                txn.put("length".encode("utf-8"), str(self.i).encode("utf-8"))


class LMDBImageWriter:
    def __init__(self, path, format='webp', quality=100, zfill=7) -> None:
        self.path = path
        self.format = format
        self.quality = quality
        self.zfill = zfill
        self.queue = None
        self.worker = None

    def __enter__(self):
        self.queue = Queue(maxsize=3)
        self.worker = _WriterWroker(self.path, self.format, self.quality,
                                    self.zfill, self.queue)
        self.worker.start()

    def put_images(self, tensor):
        """
        Args:
            tensor: (n, c, h, w) [0-1] tensor
        """
        self.queue.put(tensor.cpu())
        # with self.env.begin(write=True) as txn:
        #     for x in tensor:
        #         key = f"{str(self.i).zfill(self.zfill)}".encode("utf-8")
        #         x = convert(x, self.format, self.quality)
        #         txn.put(key, x)
        #         self.i += 1

    def __exit__(self, *args, **kwargs):
        self.queue.put(None)
        self.queue.close()
        self.worker.join()


class LMDBImageReader(Dataset):
    def __init__(self, path, zfill: int = 7):
        self.zfill = zfill
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(
                txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{str(index).zfill(self.zfill)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        return img

# FILE: ./data_resize_celebahq.py

import argparse
import multiprocessing
from functools import partial
from io import BytesIO
from pathlib import Path

import lmdb
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as trans_fn
from tqdm import tqdm
import os


def resize_and_convert(img, size, resample, quality=100):
    img = trans_fn.resize(img, size, resample)
    img = trans_fn.center_crop(img, size)
    buffer = BytesIO()
    img.save(buffer, format="jpeg", quality=quality)
    val = buffer.getvalue()

    return val


def resize_multiple(img,
                    sizes=(128, 256, 512, 1024),
                    resample=Image.LANCZOS,
                    quality=100):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size, resample, quality))

    return imgs


def resize_worker(img_file, sizes, resample):
    i, (file, idx) = img_file
    img = Image.open(file)
    img = img.convert("RGB")
    out = resize_multiple(img, sizes=sizes, resample=resample)

    return i, idx, out


def prepare(env,
            paths,
            n_worker,
            sizes=(128, 256, 512, 1024),
            resample=Image.LANCZOS):
    resize_fn = partial(resize_worker, sizes=sizes, resample=resample)

    # index = filename in int
    indexs = []
    for each in paths:
        file = os.path.basename(each)
        name, ext = file.split('.')
        idx = int(name)
        indexs.append(idx)

    # sort by file index
    files = sorted(zip(paths, indexs), key=lambda x: x[1])
    files = list(enumerate(files))
    total = 0

    with multiprocessing.Pool(n_worker) as pool:
        for i, idx, imgs in tqdm(pool.imap_unordered(resize_fn, files)):
            for size, img in zip(sizes, imgs):
                key = f"{size}-{str(idx).zfill(5)}".encode("utf-8")

                with env.begin(write=True) as txn:
                    txn.put(key, img)

            total += 1

        with env.begin(write=True) as txn:
            txn.put("length".encode("utf-8"), str(total).encode("utf-8"))


class ImageFolder(Dataset):
    def __init__(self, folder, exts=['jpg']):
        super().__init__()
        self.paths = [
            p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')
        ]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = os.path.join(self.folder, self.paths[index])
        img = Image.open(path)
        return img


if __name__ == "__main__":
    """
    converting celebahq images to lmdb
    """
    num_workers = 16
    in_path = 'datasets/celebahq'
    out_path = 'datasets/celebahq256.lmdb'

    resample_map = {"lanczos": Image.LANCZOS, "bilinear": Image.BILINEAR}
    resample = resample_map['lanczos']

    sizes = [256]

    print(f"Make dataset of image sizes:", ", ".join(str(s) for s in sizes))

    # imgset = datasets.ImageFolder(in_path)
    # imgset = ImageFolder(in_path)
    exts = ['jpg']
    paths = [p for ext in exts for p in Path(f'{in_path}').glob(f'**/*.{ext}')]

    with lmdb.open(out_path, map_size=1024**4, readahead=False) as env:
        prepare(env, paths, num_workers, sizes=sizes, resample=resample)

# FILE: ./templates_cls.py

from templates import *


def ffhq128_autoenc_cls():
    conf = ffhq128_autoenc_130M()
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = ManipulateMode.celebahq_all
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f'checkpoints/{ffhq128_autoenc_130M().name}/latent.pkl'
    conf.batch_size = 32
    conf.lr = 1e-3
    conf.total_samples = 300_000
    # use the pretraining trick instead of contiuning trick
    conf.pretrain = PretrainConfig(
        '130M',
        f'checkpoints/{ffhq128_autoenc_130M().name}/last.ckpt',
    )
    conf.name = 'ffhq128_autoenc_cls'
    return conf


def ffhq256_autoenc_cls():
    '''We first train the encoder on FFHQ dataset then use it as a pretrained to train a linear classifer on CelebA dataset with attribute labels'''
    conf = ffhq256_autoenc()
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = ManipulateMode.celebahq_all
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f'checkpoints/{ffhq256_autoenc().name}/latent.pkl'  # we train on Celeb dataset, not FFHQ
    conf.batch_size = 32
    conf.lr = 1e-3
    conf.total_samples = 300_000
    # use the pretraining trick instead of contiuning trick
    conf.pretrain = PretrainConfig(
        '130M',
        f'checkpoints/{ffhq256_autoenc().name}/last.ckpt',
    )
    conf.name = 'ffhq256_autoenc_cls'
    return conf

# FILE: ./ro_configs/__init__.py


# FILE: ./ro_configs/base.py

CONFIG = {
    # Random seed for reproducibility
    "random_seed": 42,

    # Riemannian optimization parameters
    "ro_SNR": 50, #SNR at which Riemannian optimization takes place
    "reg_lambda": 1e-5,
    "riemannian_steps": 2,
    "riemannian_lr_init": 1e-4,
    
    # Optimizer selection:
    "optimizer_type": "gradient_descent",  # Choices:["gradient_descent", "trust_region"]

    # Trust-region parameters
    "trust_region_delta0": 0.1,
    "trust_region_eta_success": 0.75,
    "trust_region_eta_fail": 0.25,
    "trust_region_gamma_inc": 2.0,
    "trust_region_gamma_dec": 0.5,

    # Line search parameters (used by gradient descent branch)
    "line_search": "strong_wolfe",
    "wolfe_c1": 1e-4,
    "wolfe_c2": 0.7,
    "max_bracket": 5,
    "max_zoom": 10,
    "max_alpha": 300,
    "armijo_rho": 1e-6,
    "armijo_beta": 0.1,

    # Retraction operator options: "identity" or "denoiser"
    "retraction_operator": "denoiser",

    # Momentum settings (if used in gradient descent)
    "use_momentum": False,  # (set to False for now)
    "momentum_coeff": 0.6,

    # Settings for fast calculation of Riemannian gradient via CG
    "cg_preconditioner": 'diagonal',
    "cg_precond_diag_samples": 50, 
    "cg_tol": 1e-6, 
    "cg_max_iter": 50,

    # Logging
    "log_dir": "logs",
    "plot_filename": "combined_plot.png",
}

# FILE: ./data_resize_ffhq.py

import argparse
import multiprocessing
from functools import partial
from io import BytesIO
from pathlib import Path

import lmdb
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as trans_fn
from tqdm import tqdm
import os


def resize_and_convert(img, size, resample, quality=100):
    img = trans_fn.resize(img, size, resample)
    img = trans_fn.center_crop(img, size)
    buffer = BytesIO()
    img.save(buffer, format="jpeg", quality=quality)
    val = buffer.getvalue()

    return val


def resize_multiple(img,
                    sizes=(128, 256, 512, 1024),
                    resample=Image.LANCZOS,
                    quality=100):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size, resample, quality))

    return imgs


def resize_worker(img_file, sizes, resample):
    i, (file, idx) = img_file
    img = Image.open(file)
    img = img.convert("RGB")
    out = resize_multiple(img, sizes=sizes, resample=resample)

    return i, idx, out


def prepare(env,
            paths,
            n_worker,
            sizes=(128, 256, 512, 1024),
            resample=Image.LANCZOS):
    resize_fn = partial(resize_worker, sizes=sizes, resample=resample)

    # index = filename in int
    indexs = []
    for each in paths:
        file = os.path.basename(each)
        name, ext = file.split('.')
        idx = int(name)
        indexs.append(idx)

    # sort by file index
    files = sorted(zip(paths, indexs), key=lambda x: x[1])
    files = list(enumerate(files))
    total = 0

    with multiprocessing.Pool(n_worker) as pool:
        for i, idx, imgs in tqdm(pool.imap_unordered(resize_fn, files)):
            for size, img in zip(sizes, imgs):
                key = f"{size}-{str(idx).zfill(5)}".encode("utf-8")

                with env.begin(write=True) as txn:
                    txn.put(key, img)

            total += 1

        with env.begin(write=True) as txn:
            txn.put("length".encode("utf-8"), str(total).encode("utf-8"))


class ImageFolder(Dataset):
    def __init__(self, folder, exts=['jpg']):
        super().__init__()
        self.paths = [
            p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')
        ]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = os.path.join(self.folder, self.paths[index])
        img = Image.open(path)
        return img


if __name__ == "__main__":
    """
    converting ffhq images to lmdb
    """
    num_workers = 16
    # original ffhq data path
    in_path = 'datasets/ffhq'
    # target output path
    out_path = 'datasets/ffhq.lmdb'

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    resample_map = {"lanczos": Image.LANCZOS, "bilinear": Image.BILINEAR}
    resample = resample_map['lanczos']

    sizes = [256]

    print(f"Make dataset of image sizes:", ", ".join(str(s) for s in sizes))

    # imgset = datasets.ImageFolder(in_path)
    # imgset = ImageFolder(in_path)
    exts = ['jpg']
    paths = [p for ext in exts for p in Path(f'{in_path}').glob(f'**/*.{ext}')]
    # print(paths[:10])

    with lmdb.open(out_path, map_size=1024**4, readahead=False) as env:
        prepare(env, paths, num_workers, sizes=sizes, resample=resample)

# FILE: ./dist_utils.py

from typing import List
from torch import distributed


def barrier():
    if distributed.is_initialized():
        distributed.barrier()
    else:
        pass


def broadcast(data, src):
    if distributed.is_initialized():
        distributed.broadcast(data, src)
    else:
        pass


def all_gather(data: List, src):
    if distributed.is_initialized():
        distributed.all_gather(data, src)
    else:
        data[0] = src


def get_rank():
    if distributed.is_initialized():
        return distributed.get_rank()
    else:
        return 0


def get_world_size():
    if distributed.is_initialized():
        return distributed.get_world_size()
    else:
        return 1


def chunk_size(size, rank, world_size):
    extra = rank < size % world_size
    return size // world_size + extra
# FILE: ./run_bedroom128_ddim.py

from templates import *
from templates_latent import *

if __name__ == '__main__':
    gpus = [0, 1, 2, 3]
    conf = bedroom128_ddpm()
    train(conf, gpus=gpus)

    gpus = [0, 1, 2, 3]
    conf.eval_programs = ['fid10']
    train(conf, gpus=gpus, mode='eval')
# FILE: ./ro_optimization/riemannian_optimization.py

#!/usr/bin/env python
"""
ro_optimization/riemannian_optimization.py

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

def get_opt_fn(cls_model, cls_id, latent_shape, x0_flat, l2_lambda=0.1):
    """
    Returns an optimization objective function that:
      - Maximizes the classifier output for a given class (minimizes its negative),
      - Encourages staying close to the original latent via L2 regularization.

    Args:
        cls_model: The classifier model.
        cls_id: The target class ID to maximize.
        latent_shape: Shape of the latent tensor (excluding batch dim).
        x0_flat: The original latent (flattened).
        l2_lambda: Weight of the L2 regularization term.

    Returns:
        opt_fn(x_flat): A callable that computes the total objective.
    """
    def opt_fn(x_flat):
        cls_loss = classifier_objective(x_flat, cls_model, cls_id, latent_shape)
        l2_loss = F.mse_loss(x_flat, x0_flat, reduction='none').sum(dim=1)
        return cls_loss + l2_lambda * l2_loss
    
    return opt_fn


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

def visualize_trajectory(model, xT, trajectory, latent_shape, T_render, save_path, fast_mode=False):
    """
    Visualizes the optimization trajectory.

    For each latent in the trajectory, renders an image using model.render(xT, latent, T=T_render).
    Arranges the images in a grid: each row corresponds to one example in the batch,
    and each column corresponds to one trajectory step (with the original on the left and final manipulated on the right).

    Args:
        model: The LitModel instance used for rendering.
        xT: The stochastic latent sample used for rendering (tensor of shape (B, C, H, W)).
        trajectory: A list of flattened latent vectors (one per optimization step).
        latent_shape: The shape of the latent (excluding the batch dimension).
        T_render: The T value used in model.render.
        save_path: File path where the figure is saved.
        fast_mode: If True, renders all trajectory images in a single batched call.
    """
    n_steps = len(trajectory)
    batch_size = trajectory[0].size(0)  # assuming all trajectory points have the same batch size
    rendered_images = []  # list of length n_steps, each element is a (B, H, W, C) numpy array

    if not fast_mode:
        # Render each trajectory step individually.
        for step in range(n_steps):
            latent_flat = trajectory[step]
            latent_unflat = unflatten_tensor(latent_flat, latent_shape)
            imgs = model.render(xT, latent_unflat, T=T_render)
            imgs = (imgs + 1) / 2.0  # normalize to [0, 1]
            # Convert from (B, C, H, W) to (B, H, W, C)
            imgs_np = imgs.cpu().permute(0, 2, 3, 1).numpy()
            rendered_images.append(imgs_np)
    else:
        # Fast mode: batch all latent steps together.
        # Unflatten and concatenate all latent steps along the batch dimension.
        all_latents = torch.cat([unflatten_tensor(latent, latent_shape) for latent in trajectory], dim=0)
        # Repeat xT for each trajectory step.
        xT_repeated = xT.repeat(n_steps, 1, 1, 1)
        imgs = model.render(xT_repeated, all_latents, T=T_render)
        imgs = (imgs + 1) / 2.0  # normalize to [0, 1]
        # Reshape: from (n_steps*B, C, H, W) to (n_steps, B, C, H, W)
        imgs = imgs.view(n_steps, batch_size, *imgs.shape[1:])
        # Permute to (n_steps, B, H, W, C)
        imgs = imgs.permute(0, 1, 3, 4, 2).cpu().numpy()
        # Split into list per trajectory step.
        for step in range(n_steps):
            rendered_images.append(imgs[step])
    
    # Create a grid: rows = batch_size, columns = n_steps.
    fig, axes = plt.subplots(batch_size, n_steps, figsize=(n_steps * 3, batch_size * 3))
    # Ensure axes is a 2D array.
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
    # Get the combined classifier + proximity objective
    l2_lambda = riem_config.get("l2_lambda", 0.2)# Retrieve lambda for L2 penalty from config (or default to 0.1)
    opt_fn = get_opt_fn(cls_model, cls_id, latent_shape, x0_flat, l2_lambda)

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
    output_dir = "ro_results"
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

    # --- Visualize the optimization trajectory ---
    traj_save_path = os.path.join(output_dir, "trajectory.png")
    # Set fast_mode to False for individual rendering; set to True to use batched rendering.
    visualize_trajectory(model, xT, trajectory, latent_shape, T_render, traj_save_path, fast_mode=False)
    
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

# FILE: ./ro_optimization/__init__.py

