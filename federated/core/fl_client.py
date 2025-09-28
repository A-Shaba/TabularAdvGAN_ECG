# federated/core/fl_client.py
import copy
import torch
from torch.utils.data import DataLoader
from data.dataset import ECGTabularDataset
from models.ecg_mlp import ECGMLP, DeepECGMLP

def make_model(name, in_dim, num_classes):
    if name == "mlp": return ECGMLP(in_dim, num_classes)
    if name == "deep_mlp": return DeepECGMLP(in_dim, num_classes)
    raise ValueError(f"Unknown model: {name}")

class FLClient:
    """
    Federated client for tabular ECG data.
    - csv_path: path to client's local CSV (train split for that client)
    - feature_cols_path / scaler_path: passed to ECGTabularDataset
    - attack_hook: optional callable(model, x, y) -> (x_mod, y_mod)
    """
    def __init__(self, cid, cfg, csv_path, feature_cols_path, scaler_path=None, attack_hook=None, device=None):
        self.cid = cid
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() and cfg["train"]["device"]=="cuda" else "cpu")

        # build dataset and dataloader
        self.ds = ECGTabularDataset(
            csv_path,
            feature_cols_path=feature_cols_path,
            scaler_path=scaler_path,
            label_col=cfg["data"].get("label_col", "Diagnosis")
        )
        self.classes = list(sorted(set(self.ds.y.tolist())))
        in_dim = len(self.ds.feature_cols)
        num_classes = len(self.classes)

        self.dl = DataLoader(
            self.ds,
            batch_size=cfg["fl"].get("client_bs", cfg["train"].get("bs", 64)),
            shuffle=True,
            num_workers=cfg["fl"].get("num_workers", 2),
            pin_memory=(self.device=="cuda")
        )

        # local model (initialized empty — weights set by server before local_train)
        self.model = make_model(cfg["model"]["name"], in_dim, num_classes).to(self.device)
        self.attack_hook = attack_hook

    def set_weights(self, state_dict):
        """Load global state into local model."""
        self.model.load_state_dict(state_dict, strict=True)

    def get_weights(self):
        """Return a deep copy of model state dict for aggregation."""
        return copy.deepcopy(self.model.state_dict())

    def local_train(self):
        """Perform local training on client's data (in-place modifies self.model)."""
        self.model.train()
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg["fl"].get("client_lr", 1e-3))
        loss_fn = torch.nn.CrossEntropyLoss()

        local_epochs = self.cfg["fl"].get("local_epochs", 1)
        for _ in range(local_epochs):
            for batch in self.dl:
                x = batch["features"].to(self.device, non_blocking=True)
                y = batch["label"].to(self.device, non_blocking=True)

                # allow attack_hook to modify batch (for malicious clients)
                if self.attack_hook is not None:
                    x, y = self.attack_hook(self.model, x, y)

                opt.zero_grad(set_to_none=True)
                logits = self.model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()
