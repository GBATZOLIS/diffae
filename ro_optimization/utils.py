import torch

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
