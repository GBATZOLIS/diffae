import torch
import torch.nn.functional as F
from .utils import unflatten_tensor

def classifier_objective(x_flat, cls_model, cls_id, latent_shape):
    """
    Computes the negative logit for the target class to maximize classifier output.
    """
    x = unflatten_tensor(x_flat, latent_shape)
    logits = cls_model.classifier(x)
    return -logits[:, cls_id]

def get_opt_fn(cls_model, cls_id, latent_shape, x0_flat, l2_lambda=0.1):
    """
    Returns an optimization objective function that combines the classifier loss with an L2 regularization term.
    """
    def opt_fn(x_flat):
        cls_loss = classifier_objective(x_flat, cls_model, cls_id, latent_shape)
        l2_loss = F.mse_loss(x_flat, x0_flat, reduction='none').sum(dim=1)
        return cls_loss + l2_lambda * l2_loss
    return opt_fn
