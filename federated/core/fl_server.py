# federated/core/fl_server.py
import copy
import torch
from torch.utils.data import DataLoader
from data.dataset import ECGTabularDataset
from models.ecg_mlp import ECGMLP, DeepECGMLP

def make_model(name, in_dim, num_classes):
    if name == "mlp": return ECGMLP(in_dim, num_classes)
    if name == "deep_mlp": return DeepECGMLP(in_dim, num_classes)
    raise ValueError(f"Unknown model: {name}")

def make_eval_dataloader(cfg):
    ds_te = ECGTabularDataset(
        cfg["data"]["test_csv"],
        feature_cols_path=cfg["data"]["feature_cols_path"],
        scaler_path=cfg["data"].get("scaler")
    )
    dl = DataLoader(ds_te,
                    batch_size=cfg["fl"].get("eval_bs", 128),
                    shuffle=False,
                    num_workers=cfg["fl"].get("num_workers", 2),
                    pin_memory=(cfg["train"].get("device","cpu")=="cuda"))
    return ds_te, dl

def fedavg(state_dicts, weights=None):
    """Simple weighted average over state dicts (list of dicts)."""
    avg = copy.deepcopy(state_dicts[0])
    n = len(state_dicts)
    if weights is None:
        weights = [1.0 / n] * n
    for k in avg.keys():
        avg[k].zero_()
        for sd, w in zip(state_dicts, weights):
            avg[k] += sd[k].float() * float(w)
    return avg

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for b in dataloader:
            x = b["features"].to(device, non_blocking=True)
            y = b["label"].to(device, non_blocking=True)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0

class FLServer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = "cuda" if (torch.cuda.is_available() and cfg["train"]["device"]=="cuda") else "cpu"

        # prepare test dataloader and determine num_classes / feature dim
        ds_te, dl_te = make_eval_dataloader(cfg)
        self.dl_te = dl_te
        self.num_classes = len(set(ds_te.y.tolist()))
        self.in_dim = len(ds_te.feature_cols)

        # initialize global model
        self.global_model = make_model(cfg["model"]["name"], self.in_dim, self.num_classes).to(self.device)

    def broadcast(self):
        """Return a copy of global state dict to be sent to clients."""
        return copy.deepcopy(self.global_model.state_dict())

    def aggregate(self, client_state_dicts, client_weights=None):
        """Aggregate client models (state_dicts) and update global model."""
        new_sd = fedavg(client_state_dicts, client_weights)
        # ensure types & device compatibility
        self.global_model.load_state_dict(new_sd, strict=True)

    def evaluate_global(self):
        """Return accuracy of the current global model on server test set."""
        return evaluate_model(self.global_model, self.dl_te, self.device)
