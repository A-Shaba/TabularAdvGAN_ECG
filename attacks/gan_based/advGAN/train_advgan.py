# training/train_advgan.py (patched for targeted & untargeted)
import argparse
import csv
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

from attacks.gan_based.advGAN.discriminator import AdvGANDiscriminatorTabular
from attacks.gan_based.advGAN.generator import AdvGANGeneratorTabular
from data.dataset import ECGTabularDataset
from models.ecg_mlp import ECGMLP, DeepECGMLP
from attacks.gan_based.advGAN.advgan import (
    AdvGANWrapperTabular,
    advgan_losses_tabular,
)

# -------------------------
# Training script for AdvGAN on tabular ECG dataset
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config with advgan params")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Load config
    # -------------------------
    cfg = yaml.safe_load(open(args.config))

    train_csv = cfg["data"]["train_csv"]
    val_csv = cfg["data"]["val_csv"]
    feature_cols_path = cfg["data"]["feature"]
    scaler_path = cfg["data"]["scaler"]

    # -------------------------
    # Dataset
    # -------------------------

    train_ds = ECGTabularDataset(train_csv, feature_cols_path,
                                 scaler_path)
    val_ds = ECGTabularDataset(val_csv, feature_cols_path,
                               scaler_path)

    dl_train = DataLoader(train_ds, batch_size=cfg["train"]["bs"], shuffle=True)
    dl_val = DataLoader(val_ds, batch_size=cfg["train"]["bs"])

    in_dim = len(train_ds.feature_cols)
    num_classes = len(set(train_ds.y.tolist()))
    print(f"[INFO] Input dimension: {in_dim}, Num classes: {num_classes}")
    # -------------------------
    # Target model (frozen)
    # -------------------------
    if cfg["model"]["name"] == "mlp":
        target = ECGMLP(in_dim, num_classes)
    elif cfg["model"]["name"] == "deep_mlp":
        target = DeepECGMLP(in_dim, num_classes)
    else:
        raise ValueError(f"Unknown model {cfg['model']['name']}")
    
    
    target.load_state_dict(torch.load(cfg["model"]["ckpt"]))
    target.to(device)
    target.eval()



    # -------------------------
    # Generator / Discriminator
    # -------------------------
    G = AdvGANGeneratorTabular(in_dim)
    D = AdvGANDiscriminatorTabular(in_dim)
    G.to(device); D.to(device)

    wrap = AdvGANWrapperTabular(target, G, D, eps=cfg["attack"]["advgan"]["eps"]).to(device)

    g_opt = optim.Adam(G.parameters(), lr=cfg["train"]["g_lr"])
    d_opt = optim.Adam(D.parameters(), lr=cfg["train"]["d_lr"])

    # -------------------------
    # Attack parameters
    # -------------------------
    adv_cfg = cfg["attack"]["advgan"]
    targeted = adv_cfg.get("targeted", False)
    target_class = adv_cfg.get("target_class", None)
    attack_loss_type = adv_cfg.get("attack_loss_type", "ce")
    kappa = float(adv_cfg.get("kappa", 0.0))


    #salvataggi
    out_dir = Path(cfg["advgan"].get("out_dir", "outputs/advgan"))
    out_g = out_dir / "generator.pt"
    out_d = out_dir / "discriminator.pt"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.csv"

    # Overwrite header
    with log_path.open("w", newline="") as f:
        csv.writer(f).writerow(["epoch", "gen_loss", "disc_loss", "asr"])

    print(f"[INFO] Starting training for {cfg['train']['epochs']} epochs...")

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(cfg["train"]["epochs"]):
        G.train(); D.train()
        g_loss_total, d_loss_total = 0.0, 0.0

        for b in dl_train:
            x, y = b["features"].to(device), b["label"].to(device)
            y_t = None
            if targeted:
                if target_class is None:
                    raise ValueError("target_class must be set when targeted=True")
                y_t = torch.full_like(y, fill_value=int(target_class), device=device)

            # ----- Discriminator -----
            for _ in range(cfg["train"]["d_steps"]):
                d_opt.zero_grad()
                x_adv, _ = wrap.perturb(x)
                g_loss_batch, d_loss_batch = advgan_losses_tabular(
                    D, target, x, y, x_adv,
                    lambda_adv=adv_cfg["lambda_adv"],
                    lambda_gan=adv_cfg.get("lambda_gan", 1.0),
                    lambda_pert=adv_cfg.get("lambda_pert", 0.01),
                    targeted=targeted,
                    y_target=y_t,
                    attack_loss_type=attack_loss_type,
                    kappa=kappa
                )
                d_loss_batch.backward()
                d_opt.step()
                d_loss_total += d_loss_batch.item()

            # ----- Generator -----
            for _ in range(cfg["train"]["g_steps"]):
                g_opt.zero_grad()
                x_adv, _ = wrap.perturb(x)
                g_loss_batch, _ = advgan_losses_tabular(
                    D, target, x, y, x_adv,
                    lambda_adv=adv_cfg["lambda_adv"],
                    lambda_gan=adv_cfg.get("lambda_gan", 1.0),
                    lambda_pert=adv_cfg.get("lambda_pert", 0.01),
                    targeted=targeted,
                    y_target=y_t,
                    attack_loss_type=attack_loss_type,
                    kappa=kappa
                )
                g_loss_batch.backward()
                g_opt.step()
                g_loss_total += g_loss_batch.item()

        # Evaluate attack success
        asr = wrap.attack_success_rate(
            dl_val,
            targeted=targeted,
            target_class=(int(target_class) if targeted else None),
            device=device
        )
        g_loss_avg = g_loss_total / len(dl_train)
        d_loss_avg = d_loss_total / len(dl_train)

        # Append to CSV log immediately
        with log_path.open("a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                g_loss_avg,
                d_loss_avg,
                asr
            ])

        mode = "Targeted" if targeted else "Untargeted"
        print(f"[{epoch+1}/{cfg['train']['epochs']}] "
              f"G_loss={g_loss_avg:.4f}, D_loss={d_loss_avg:.4f}, "
              f"{mode} ASR={asr:.3f}")

    # -------------------------
    # Save generator
    # -------------------------
    torch.save(G.state_dict(), out_g)
    torch.save(D.state_dict(), out_d)
    print(f"[INFO] Saved AdvGAN models and logs to {out_dir}")


if __name__ == "__main__":
    main()
