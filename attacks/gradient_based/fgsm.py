# attacks/gradient_based/fgsm.py
import torch
import torch.nn.functional as F

def fgsm_attack(model, x, y, eps=0.05, clip_min=None, clip_max=None):
    """
    FGSM for tabular features.
    model: classifier; x: (N, in_dim) float tensor; y: (N,) long
    Returns x_adv with same shape.
    """
    x_adv = x.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    with torch.no_grad():
        x_adv = x_adv + eps * x_adv.grad.sign()
        if clip_min is not None or clip_max is not None:
            x_adv = torch.clamp(x_adv, min=clip_min, max=clip_max)
    return x_adv.detach()
