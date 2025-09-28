# reset_project.py
import shutil
from pathlib import Path

def remove_dir_safe(path: Path):
    if path.exists() and path.is_dir():
        print(f"[INFO] Removing {path}")
        shutil.rmtree(path)
    else:
        print(f"[INFO] Skipping {path}, not found or not a directory.")

def remove_file_safe(path: Path):
    if path.exists() and path.is_file():
        print(f"[INFO] Removing {path}")
        path.unlink()
    else:
        print(f"[INFO] Skipping {path}, not found or not a file.")

def reset_project():
    """
    Cancella tutti i dati generati dai vari step:
    - output del preprocessing (train/val/test CSV, scaler)
    - output dei modelli (baseline, AdvGAN, federated)
    - plot generati per la webapp
    """
    # --- Processed dataset ---
    proc_dir = Path("data/processed")
    remove_dir_safe(proc_dir)

    # --- Baseline outputs ---
    baseline_dir = Path("outputs/baseline")
    remove_dir_safe(baseline_dir)

    # --- AdvGAN outputs ---
    advgan_dir = Path("outputs/advgan")
    remove_dir_safe(advgan_dir)

    # --- Federated outputs ---
    fed_dir = Path("outputs/federated")
    remove_dir_safe(fed_dir)

    # --- Webapp plots ---
    plots_dir = Path("webapp/static/plots")
    remove_dir_safe(plots_dir)

    # --- Optional: temp logs ---
    logs_dir = Path("outputs/logs")
    remove_dir_safe(logs_dir)

    print("[DONE] Project reset completed.")

if __name__ == "__main__":
    reset_project()
