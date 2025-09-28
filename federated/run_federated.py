# federated/run_federated.py
import argparse, yaml, torch, csv
from pathlib import Path
from federated.core.fl_server import FLServer
from federated.core.fl_client import FLClient
from federated.fl_attack_wrapper import PoisonWithAdvGAN
from federated.split_clients import split_clients
from models.ecg_mlp import ECGMLP, DeepECGMLP

def make_target(name, in_dim, num_classes):
    if name == "mlp": return ECGMLP(in_dim, num_classes)
    if name == "deep_mlp": return DeepECGMLP(in_dim, num_classes)
    raise ValueError(name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    client_paths = [Path(p) for p in cfg["fl"]["client_csvs"]]
    if not all(p.exists() for p in client_paths):
        split_clients(cfg["data"]["train_csv"], Path(cfg["fl"]["out_dir"]))
        client_paths = [Path(cfg["fl"]["out_dir"]) / f"client{i}.csv" for i in range(len(client_paths))]

    server = FLServer(cfg)
    device = server.device

    # Build clients
    clients = []
    attacker_ref = make_target(cfg["model"]["name"], server.in_dim, server.num_classes).to(device)
    attacker_ref.load_state_dict(torch.load(cfg["model"]["ckpt"], map_location=device)); attacker_ref.eval()
    for p in attacker_ref.parameters(): p.requires_grad=False

    adv_ckpt = cfg["attack"]["advgan"]["model_ckpt"]
    adv_eps  = cfg["attack"]["advgan"].get("eps", 0.05)
    poison_frac = cfg["fl"].get("poison_frac", 0.5)
    malicious_index = cfg["fl"].get("malicious_index", 1)
    poison_hook = PoisonWithAdvGAN(attacker_ref, adv_ckpt, adv_eps, poison_frac, device)

    feature_cols_path = cfg["data"]["feature_cols_path"]
    scaler_path = cfg["data"].get("scaler", None)

    for i, csv_path in enumerate(client_paths):
        hook = poison_hook if i==malicious_index else None
        print(f"[INFO] Client {i} using CSV: {csv_path} {'[MALICIOUS]' if hook else ''}")
        clients.append(FLClient(i, cfg, csv_path, feature_cols_path, scaler_path, attack_hook=hook, device=device))

    rounds = cfg["fl"].get("rounds",5)
    out_dir = Path(cfg["fl"].get("out_dir","outputs/federated"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # OVERWRITE CSV at start
    acc_log_path = out_dir / "round_accuracies.csv"
    with acc_log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round","global_acc","clients"])

    # Federated loop
    for r in range(rounds):
        print(f"\n=== Federated Round {r+1}/{rounds} ===")
        global_sd = server.broadcast()
        for c in clients: c.set_weights(global_sd)
        for c in clients:
            print(f"Client {c.cid} local training...")
            c.local_train()
        server.aggregate([c.get_weights() for c in clients])
        acc = server.evaluate_global()
        print(f"[Round {r+1}] global clean accuracy = {acc:.4f}")

        with acc_log_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([r+1, acc, len(clients)])

    torch.save(server.global_model.state_dict(), out_dir/"global_model.pt")
    print(f"[INFO] Saved federated logs and global model to {out_dir}")

if __name__=="__main__":
    main()
