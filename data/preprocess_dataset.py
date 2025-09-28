# data/preprocess_dataset.py  (tabular splitter + automatic scaler)
import argparse
from pathlib import Path
import pandas as pd
import json
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

def fit_scaler_on_df(df, feature_cols, out_path):
    stats = {}
    for c in feature_cols:
        vals = df[c].values.astype(np.float32)
        stats[c] = {"mean": float(vals.mean()), "std": float(vals.std() + 1e-8)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved scaler to {out_path}")

def save_feature_cols(feature_cols, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"Saved feature column order to {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True, help="Path to full CSV with features + label")
    ap.add_argument("--output-dir", required=True, help="Where to write train/val/test CSVs and scaler")
    ap.add_argument("--label-col", default="Diagnosis")
    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--test-size", type=float, default=0.1)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--no-scaler", action="store_true", help="Do not compute/save scaler")
    ap.add_argument("--force", action="store_true", help="Overwrite existing scaler/file if present")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    label_col = args.label_col

    # determine feature columns (exclude label)
    feature_cols = [c for c in df.columns if c != label_col]

    # Stratified split -> train / test
    y = df[label_col].values
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    train_idx, test_idx = next(sss1.split(df, y))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # split train -> train + val
    val_ratio = args.val_size / (1.0 - args.test_size)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=args.random_state)
    tr_idx, val_idx = next(sss2.split(df_train, df_train[label_col].values))
    df_tr = df_train.iloc[tr_idx].reset_index(drop=True)
    df_val = df_train.iloc[val_idx].reset_index(drop=True)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_path = outdir / "train.csv"
    val_path = outdir / "val.csv"
    test_path = outdir / "test.csv"
    scaler_path = outdir / "scaler.json"
    featcols_path = outdir / "feature_cols.json"

    df_tr.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)
    print("Wrote:", train_path, val_path, test_path)

    # save feature columns (ensures consistent ordering later)
    if featcols_path.exists() and not args.force:
        print(f"{featcols_path} exists (use --force to overwrite).")
    else:
        save_feature_cols(feature_cols, featcols_path)

    # fit scaler on training set and save
    if args.no_scaler:
        print("Skipping scaler creation (--no-scaler).")
    else:
        if scaler_path.exists() and not args.force:
            print(f"{scaler_path} exists (use --force to overwrite).")
        else:
            print("Fitting scaler on training set...")
            fit_scaler_on_df(df_tr, feature_cols, scaler_path)

if __name__ == "__main__":
    main()
