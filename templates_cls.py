from templates import *
from templates_latent import *

def ffhq128_autoenc_cls():
    conf = ffhq128_autoenc_130M()
    conf.autoenc_config = ffhq128_autoenc_latent()
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
    #settings for diffusion-time dependent classifiers
    conf.diffusion_time_dependent_classifier=False
    conf.lower_trainable_snr = 1
    return conf

def ffhq128_autoenc_non_linear_cls():
    conf = ffhq128_autoenc_130M()
    conf.autoenc_config = ffhq128_autoenc_latent()
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = ManipulateMode.celebahq_all
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f'checkpoints/{ffhq128_autoenc_130M().name}/latent.pkl'
    conf.batch_size = 64
    conf.lr = 5e-4
    conf.total_samples = 300_000
    # Use the pretraining trick instead of continuing training.
    conf.pretrain = PretrainConfig(
        '130M',
        f'checkpoints/{ffhq128_autoenc_130M().name}/last.ckpt',
    )
    # Set classifier type to 'nonlinear' and define extra parameters.
    conf.classifier_type = 'nonlinear'
    conf.non_linear_hidden_dims = [256]
    conf.non_linear_dropout = 0.1
    conf.name = 'ffhq128_autoenc_cls_nonlinear'
    #settings for diffusion-time dependent classifiers
    conf.diffusion_time_dependent_classifier=False
    conf.lower_trainable_snr = 1
    return conf

def ffhq128_autoenc_non_linear_time_cls():
    conf = ffhq128_autoenc_non_linear_cls()
    #conf.autoenc_config.T_eval = 1000
    conf.autoenc_config.latent_T_eval = 1000
    conf.name = 'ffhq128_autoenc_time_cls_nonlinear'
    conf.diffusion_time_dependent_classifier = True
    conf.lower_trainable_snr = 1
    return conf

def ffhq128_autoenc_non_linear_time_cls_full():
    conf = ffhq128_autoenc_non_linear_cls()
    #conf.autoenc_config.T_eval = 1000
    conf.autoenc_config.latent_T_eval = 1000
    conf.name = 'ffhq128_autoenc_time_cls_nonlinear_full'
    conf.diffusion_time_dependent_classifier = True
    conf.lower_trainable_snr = 1e-2
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
