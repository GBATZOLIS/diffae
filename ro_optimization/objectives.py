import torch
import torch.nn.functional as F

def get_opt_fn(classifier_fn, cls_id, latent_shape, x0_flat, classifier_weight=1., reg_norm_weight=0.1, reg_norm_type="L2"):
    """
    Returns an optimization objective function that combines a classifier loss evaluated
    at a specific diffusion time (via classifier_fn) with a regularization term that can be either L2 or L1.
    
    Args:
        classifier_fn: A function such that classifier_fn(x_flat) returns raw logits 
                       when given a flattened latent.
        cls_id: The target classifier output id.
        latent_shape: The original shape of the latent.
        x0_flat: The original flattened latent (used for computing the regularization term).
        reg_norm_weight: The weight applied to the regularization term.
        reg_norm_type: Type of norm to use for regularization. Should be either "L2" (default) or "L1".
        
    Returns:
        opt_fn: A function that maps a flattened latent x_flat to a per-sample scalar objective value.
                The objective is the sum of the negative logit for the target class (to be maximized)
                and the weighted norm difference between x_flat and x0_flat.
    """
    def opt_fn(x_flat):
        logits = classifier_fn(x_flat)
        cls_loss = -logits[:, cls_id]  # We want to maximize the logit for the target class.
        if reg_norm_type.upper() == "L2":
            reg_loss = F.mse_loss(x_flat, x0_flat, reduction='none').sum(dim=1)
        elif reg_norm_type.upper() == "L1":
            reg_loss = F.l1_loss(x_flat, x0_flat, reduction='none').sum(dim=1)
        else:
            raise ValueError(f"Unknown norm type: {reg_norm_type}. Use 'L2' or 'L1'.")
        return classifier_weight * cls_loss + reg_norm_weight * reg_loss
    return opt_fn
