import argparse
import yaml
import torch
import json
from pathlib import Path
from torch.utils.data import DataLoader

from data.dataset import ECGTabularDataset
from models.ecg_mlp import ECGMLP, DeepECGMLP
from attacks.gradient_based.fgsm import fgsm_attack
from attacks.gradient_based.pgd import pgd_attack
from attacks.gan_based.advGAN.attack_advgan import AdvGAN_Attack
from utils.metrics import accuracy

def make_model(name, in_dim, num_classes):
    if name == "mlp":
        return ECGMLP(in_dim, num_classes)
    if name == "deep_mlp":
        return DeepECGMLP(in_dim, num_classes)
    raise ValueError(f"Unsupported model name: {name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config YAML file")
    ap.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    ap.add_argument("--attack", default="none", choices=["none", "fgsm", "pgd", "advgan"], help="Type of attack to evaluate")
    args = ap.parse_args()

    # Check file paths
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    if not Path(args.ckpt).exists():
        raise FileNotFoundError(f"Checkpoint file not found: {args.ckpt}")

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Load test dataset
    ds_te = ECGTabularDataset(
        cfg["data"]["test_csv"],
        feature_cols_path=cfg["data"]["feature"],
        scaler_path=cfg["data"]["scaler"]
    )
    in_dim = len(ds_te.feature_cols)
    num_classes = len(set(ds_te.y.tolist()))
    dl_te = DataLoader(ds_te, batch_size=cfg["eval"].get("bs", 128), shuffle=False)

    # Load model
    device = cfg["eval"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(cfg["model"]["name"], in_dim, num_classes).to(device)

    try:
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
    except RuntimeError as e:
        raise RuntimeError(f"Error loading model checkpoint: {e}")
    
    model.eval()

    # Compute clean accuracy and get clean prediction details
    clean_acc, clean_preds, clean_labels, clean_mask = accuracy(
        model, dl_te, device=device, return_details=True
    )

    robust_acc = None
    per_class_asr = {}

    if args.attack != "none":
        if args.attack == "fgsm":
            eps = cfg["attack"].get("fgsm_eps", 0.05)
            def adv_fn(m, x, y): return fgsm_attack(m, x, y, eps=eps)

        elif args.attack == "pgd":
            eps = cfg["attack"].get("pgd_eps", 0.05)
            alpha = cfg["attack"].get("pgd_alpha", 0.01)
            steps = cfg["attack"].get("pgd_steps", 10)
            def adv_fn(m, x, y): return pgd_attack(m, x, y, eps=eps, alpha=alpha, steps=steps)

        elif args.attack == "advgan":
            advgan_cfg = cfg["attack"].get("advgan", {})
            advgan_ckpt = advgan_cfg.get("model_ckpt", None)
            if advgan_ckpt is None:
                raise ValueError("AdvGAN generator checkpoint must be specified in config under attack.advgan.model_ckpt")
            eps = advgan_cfg.get("eps", 0.03)
            advgan_attack = AdvGAN_Attack(model, ckpt_path=advgan_ckpt, eps=eps, device=device)
            def adv_fn(m, x, y):
                return advgan_attack(x)

        else:
            raise ValueError(f"Unsupported attack type: {args.attack}")

        # Filter correctly predicted samples
        correct_indices = torch.where(clean_mask)[0]
        correct_inputs = torch.stack([ds_te[i]["features"] for i in correct_indices])
        correct_labels = clean_labels[correct_indices]

        robust_preds = []
        model.eval()
        bs = cfg["eval"].get("bs", 128)
        for i in range(0, len(correct_inputs), bs):
            x_batch = correct_inputs[i:i+bs].to(device)
            y_batch = correct_labels[i:i+bs].to(device)
            x_adv = adv_fn(model, x_batch, y_batch)
            with torch.no_grad():
                preds = model(x_adv).argmax(dim=1)
            robust_preds.append(preds.cpu())

        robust_preds = torch.cat(robust_preds)

        # Calculate per-class ASR (Attack Success Rate)
        for cls in torch.unique(correct_labels):
            cls_mask = (correct_labels == cls)
            n_total = cls_mask.sum().item()
            n_flipped = (robust_preds[cls_mask] != cls).sum().item()
            per_class_asr[int(cls)] = round(n_flipped / n_total, 4) if n_total > 0 else None

        # Compute robust accuracy on full dataset using adversary function
        robust_acc = accuracy(model, dl_te, device=device, adversary=adv_fn)

    # Save/update results JSON
    out_dir = Path(cfg["train"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_path = out_dir / "eval_results.json"

    if eval_path.exists():
        with open(eval_path, "r") as f:
            results = json.load(f)
    else:
        results = {"clean_acc": clean_acc, "attacks": {}}

    results["clean_acc"] = clean_acc
    results["attacks"][args.attack] = {
        "robust_acc": robust_acc,
        "per_class_asr": per_class_asr
    }

    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)

    # Console output
    print(f"\n[INFO] Evaluation results saved to {eval_path}")
    print(f"=> Clean Accuracy: {clean_acc:.4f}")
    if args.attack != "none":
        print(f"=> Robust Accuracy under '{args.attack}': {robust_acc:.4f}")
        print("=> Per-Class ASR:")
        for cls, asr in per_class_asr.items():
            print(f"   Class {cls}: {asr:.4f}")

if __name__ == "__main__":
    main()
