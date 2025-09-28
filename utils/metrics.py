# utils/metrics.py
import torch

def accuracy(model, dataloader, device="cpu", adversary=None, return_details=False):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels, clean_mask = [], [], []

    for batch in dataloader:
        x = batch["features"].to(device)
        y = batch["label"].to(device)

        if adversary is not None:
            x = adversary(model, x, y)

        with torch.no_grad():
            logits = model(x)
            preds = logits.argmax(dim=1)

        correct_preds = (preds == y)
        correct += correct_preds.sum().item()
        total += y.size(0)

        if return_details:
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
            clean_mask.append(correct_preds.cpu())

    acc = correct / total if total > 0 else 0.0

    if return_details:
        return (
            acc,
            torch.cat(all_preds),
            torch.cat(all_labels),
            torch.cat(clean_mask)
        )

    return acc
