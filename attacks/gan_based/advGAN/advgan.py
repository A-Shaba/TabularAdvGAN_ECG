# attacks/gan_based/advGAN/advgan.py
import torch
import torch.nn.functional as F

class AdvGANWrapperTabular(torch.nn.Module):
    def __init__(self, target_model, G, D, eps=0.05):
        super().__init__()
        self.target = target_model.eval()
        for p in self.target.parameters():
            p.requires_grad = False
        self.G, self.D, self.eps = G, D, eps

    def perturb(self, x):
        """
        Returns (x_adv, delta). Delta is bounded by tanh * eps.
        x expected already on correct device.
        """
        delta = self.G(x)
        delta = torch.tanh(delta) * self.eps
        x_adv = torch.clamp(x + delta, -1.0, 1.0)  # loose clamp; adjust to your scaled ranges
        return x_adv, delta

    @torch.no_grad()
    def attack_success_rate(self, dataloader, targeted=False, target_class=None, device=None):
        """
        ASR:
         - Untargeted: proportion of originally-correct inputs that become misclassified after attack.
         - Targeted: proportion of originally-correct inputs that are classified AS the target_class after attack.
        """
        device = device or next(self.target.parameters()).device
        correct_before = 0
        succ_after = 0
        self.target.eval(); self.G.eval()
        for batch in dataloader:
            x = batch["features"].to(device)
            y = batch["label"].to(device)
            preds_before = self.target(x).argmax(dim=1)
            # only consider those correctly classified originally
            mask = preds_before == y
            if mask.sum().item() == 0:
                continue
            x_adv, _ = self.perturb(x)
            preds_after = self.target(x_adv).argmax(dim=1)
            if targeted:
                # success if model predicts target_class on formerly-correct samples
                succ_after += (preds_after[mask] == target_class).sum().item()
            else:
                # success if model no longer predicts true label
                succ_after += (preds_after[mask] != y[mask]).sum().item()
            correct_before += mask.sum().item()

        if correct_before == 0:
            return 0.0
        return succ_after / correct_before


def cw_loss(logits, y, targeted=False, y_target=None, kappa=0.0):
    """
    Carlini-Wagner style loss. Works on logits.
      - Untargeted: maximize (best_other - correct) -> loss = clamp(correct - best_other + kappa, min=0)
      - Targeted:   maximize (target - best_other)   -> loss = clamp(other_best - target + kappa, min=0)
    Returns mean over batch (to be minimized by optimizer).
    """
    num_classes = logits.size(1)
    one_hot = F.one_hot(y, num_classes).bool()

    if targeted:
        if y_target is None:
            raise ValueError("y_target must be provided for targeted CW loss")
        # target_logit, best_other (exclude target)
        target_logit = logits.gather(1, y_target.view(-1, 1)).squeeze(1)
        mask_target = F.one_hot(y_target, num_classes).bool()
        other_logit = logits.masked_fill(mask_target, float("-inf")).max(dim=1)[0]
        loss = F.relu(other_logit - target_logit + kappa)
    else:
        correct_logit = logits.gather(1, y.view(-1, 1)).squeeze(1)
        other_logit = logits.masked_fill(one_hot, float("-inf")).max(dim=1)[0]
        loss = F.relu(correct_logit - other_logit + kappa)

    return loss.mean()


def advgan_losses_tabular(D, target_model, x, y, x_adv,
                          lambda_adv=1.0, lambda_gan=1.0, lambda_pert=0.01,
                          targeted=False, y_target=None,
                          attack_loss_type="ce", kappa=0.0):
    """
    Compute generator (g_loss) and discriminator (d_loss).
    - attack_loss_type: "ce" or "cw"
    - targeted: if True, y_target must be provided (tensor of shape (N,))
    - lambda_gan is weight for GAN fooling term
    - lambda_pert is L2 regularizer on delta (small)
    """
    # === Generator losses ===
    logits_adv = target_model(x_adv)

    if attack_loss_type == "cw":
        g_loss_cls = cw_loss(logits_adv, y, targeted=targeted, y_target=y_target, kappa=kappa)
        coeff_adv = lambda_adv
    else:
        # CE: targeted -> minimize CE wrt target (make model predict target),
        # untargeted -> maximize CE wrt true label (so we minimize -CE).
        if targeted:
            if y_target is None:
                raise ValueError("y_target must be provided for targeted CE")
            g_loss_cls = F.cross_entropy(logits_adv, y_target)
            coeff_adv = lambda_adv
        else:
            g_loss_cls = F.cross_entropy(logits_adv, y)  # normal CE
            g_loss_cls = -g_loss_cls  
            coeff_adv = -lambda_adv  # flip sign for untargeted

    # GAN loss: encourage discriminator to label x_adv as real (1)
    D_x_adv = D(x_adv)
    g_loss_gan = F.binary_cross_entropy_with_logits(D_x_adv, torch.ones_like(D_x_adv))

    # === Perturbation regularization ===
    delta = (x_adv - x).view(x.size(0), -1)
    g_loss_pert = delta.norm(p=2, dim=1).mean()

    # === Total generator loss ===
    g_loss = coeff_adv * g_loss_cls + lambda_gan * g_loss_gan + lambda_pert * g_loss_pert

    # === Discriminator loss ===
    d_real = F.binary_cross_entropy_with_logits(D(x), torch.ones_like(D(x)))
    d_fake = F.binary_cross_entropy_with_logits(D_x_adv.detach(), torch.zeros_like(D_x_adv))
    d_loss = 0.5 * (d_real + d_fake)

    return g_loss, d_loss
