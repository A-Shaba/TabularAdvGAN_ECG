# attacks/gradient_based/pgd.py
import torch
import torch.nn.functional as F

def pgd_attack(model, x, y, eps=0.05, alpha=0.01, steps=10, clip_min=None, clip_max=None):
    """
    PGD for tabular features (L_inf).
    """
    x_orig = x.detach()
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
            if clip_min is not None or clip_max is not None:
                x_adv = torch.clamp(x_adv, min=clip_min, max=clip_max)
            x_adv = x_adv.detach()
    return x_adv
