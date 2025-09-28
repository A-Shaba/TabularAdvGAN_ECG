# training/train_baseline.py
import argparse, yaml, os, csv
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from data.dataset import ECGTabularDataset
from models.ecg_mlp import ECGMLP, DeepECGMLP
from utils.metrics import accuracy
from utils.seed import seed_everything
from tqdm import tqdm

def make_model(name, in_dim, num_classes):
    if name=="mlp": return ECGMLP(in_dim, num_classes)
    if name=="deep_mlp": return DeepECGMLP(in_dim, num_classes)
    raise ValueError(name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    seed_everything(cfg.get("seed", 42))

    # load datasets
    ds_tr = ECGTabularDataset(cfg["data"]["train_csv"], feature_cols_path=cfg["data"]["feature"], scaler_path=cfg["data"]["scaler"])
    ds_va = ECGTabularDataset(cfg["data"]["val_csv"], feature_cols_path=cfg["data"]["feature"], scaler_path=cfg["data"]["scaler"])

    in_dim = len(ds_tr.feature_cols)
    num_classes = len(set(ds_tr.y.tolist()))

    device = cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(cfg["model"]["name"], in_dim, num_classes).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["train"]["epochs"])
    criterion = nn.CrossEntropyLoss()

    dl_tr = DataLoader(ds_tr, batch_size=cfg["train"]["bs"], shuffle=True, num_workers=cfg["train"].get("num_workers",2))
    dl_va = DataLoader(ds_va, batch_size=cfg["train"]["bs"], shuffle=False, num_workers=cfg["train"].get("num_workers",2))

    out_dir = Path(cfg["train"]["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir/"training_log.csv"
    
    with log_path.open("w", newline="") as f:
        csv.writer(f).writerow(["epoch","lr","train_loss","val_loss","val_acc"])

    best = 0.0
    for epoch in range(cfg["train"]["epochs"]):
        model.train(); total_loss=0.0; total_samples=0
        for b in tqdm(dl_tr, desc=f"epoch {epoch}"):
            x = b["features"].to(device)
            y = b["label"].to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward(); opt.step()
            total_loss += loss.item()*y.size(0); total_samples += y.size(0)
        sched.step()
        train_loss = total_loss/max(1,total_samples)

        # val
        model.eval(); val_loss_sum=0.0; val_n=0
        with torch.no_grad():
            for b in dl_va:
                x = b["features"].to(device)
                y = b["label"].to(device)
                logits = model(x)
                val_loss_sum += criterion(logits,y).item()*y.size(0)
                val_n += y.size(0)
        val_loss = val_loss_sum/max(1,val_n)
        val_acc = accuracy(model, dl_va, device=device)

        # log
        curr_lr = opt.param_groups[0]["lr"]
        print(f"[epoch {epoch}] lr={curr_lr:.6f} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        with log_path.open("a", newline="") as f:
            csv.writer(f).writerow([epoch,curr_lr,train_loss,val_loss,val_acc])

        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), out_dir/"baseline_best.pt")

if __name__=="__main__":
    main()
