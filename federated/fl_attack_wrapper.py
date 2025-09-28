# federated/fl_attack_wrapper.py
"""
Poisoning wrapper for federated clients using tabular AdvGAN.
Provides:
  - AdvGAN_Attack_Tabular: loads trained generator and applies perturbations
  - PoisonWithAdvGAN: client hook that replaces a fraction of a batch with adv examples
"""

import torch
from pathlib import Path

# Adjust imports to the actual location of your tabular advgan modules
from attacks.gan_based.advGAN.generator import AdvGANGeneratorTabular
from attacks.gan_based.advGAN.advgan import AdvGANWrapperTabular

class AdvGAN_Attack_Tabular:
    """
    Inference-time wrapper for tabular AdvGAN generator.

    Usage:
        atk = AdvGAN_Attack_Tabular(target_model, ckpt_path, eps=0.05, device='cuda')
        x_adv = atk(x)   # x: (N, in_dim) tensor, returns perturbed tensor on same device
    """
    def __init__(self, target_model, ckpt_path, eps=0.05, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.target = target_model.to(self.device).eval()
        for p in self.target.parameters():
            p.requires_grad = False
            
        # load checkpoint
        ckpt = Path(ckpt_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"AdvGAN generator checkpoint not found: {ckpt}")
        # load to cpu first to read shapes
        state = torch.load(str(ckpt), map_location="cpu")
        # Create generator with proper in_dim if required:
        # try to infer in_dim from first linear weight shape if generator is MLP:
        try:
            # find first linear weight param to infer in_dim
            first_w = next((v for k,v in state.items() if "net" in k and v.ndim==2), None)
            if first_w is not None:
                in_dim = first_w.shape[1]
                # re-create generator with inferred in_dim if constructor accepts it
                try:
                    self.G = AdvGANGeneratorTabular(in_dim=in_dim)
                except TypeError:
                    # fallback: generator constructor doesn't accept in_dim, use default
                    self.G = AdvGANGeneratorTabular()
            else:
                self.G = AdvGANGeneratorTabular()
        except Exception:
            self.G = AdvGANGeneratorTabular()

        # now load weights to device
        self.G.load_state_dict(state)
        self.G.to(self.device).eval()

        # wrapper (D not needed at inference; pass None)
        try:
            self.wrapper = AdvGANWrapperTabular(self.target, self.G, None, eps=eps).to(self.device)
        except TypeError:
            # if wrapper named differently or signature differs, fallback to using G directly
            self.wrapper = None
            self.eps = eps

    @torch.no_grad()
    def __call__(self, x):
        x = x.to(self.device)
        if self.wrapper is not None:
            x_adv, _ = self.wrapper.perturb(x)
            return x_adv
        else:
            # fallback: apply generator directly and scale via tanh*eps
            delta = self.G(x)
            delta = torch.tanh(delta) * self.eps
            x_adv = torch.clamp(x + delta, -10.0, 10.0)
            return x_adv


class PoisonWithAdvGAN:
    """
    Malicious client hook: replace a fraction of each batch with AdvGAN-perturbed examples.
    Keeps the original labels (error-injection poisoning).
    Constructor args:
      - target_model: the reference model (same arch used to train generator); the wrapper will load it (frozen)
      - ckpt_path: path to generator checkpoint
      - eps: perturbation magnitude (same scale used when training the generator)
      - frac: fraction of examples in each batch to poison (0..1)
      - device: device to run attack on
    """
    def __init__(self, target_model, ckpt_path, eps=0.05, frac=0.5, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.atk = AdvGAN_Attack_Tabular(target_model, ckpt_path=ckpt_path, eps=eps, device=self.device)
        self.frac = float(frac)

    def __call__(self, model, x, y):
        """
        model: local model (not used by attacker, kept for API compatibility)
        x: features tensor (N, in_dim)
        y: labels tensor (N,)
        returns: (x_mod, y_mod)
        """
        bsz = x.size(0)
        k = max(1, int(self.frac * bsz))
        # choose indices to poison
        idx = torch.randperm(bsz, device=x.device)[:k]
        x_select = x[idx]
        # generate poisoned examples
        x_poison = self.atk(x_select)
        x_new = x.clone()
        x_new[idx] = x_poison.to(x.device)
        return x_new, y
