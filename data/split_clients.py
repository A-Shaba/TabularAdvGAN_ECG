# data/split_clients.py
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

def split_clients(train_csv, out_dir, n_clients=3, random_state=42):
    outdir = Path(out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(train_csv)
    y = df["Diagnosis"]

    skf = StratifiedKFold(n_splits=n_clients, shuffle=True, random_state=random_state)
    for i, (_, idx) in enumerate(skf.split(df, y)):
        client_path = outdir / f"client{i}.csv"
        df.iloc[idx].to_csv(client_path, index=False)
        print(f"Saved {client_path} with {len(idx)} rows")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True, help="Path to train.csv")
    ap.add_argument("--out-dir", required=True, help="Output directory for client CSVs")
    ap.add_argument("--n-clients", type=int, default=3)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()
    split_clients(args.train_csv, args.out_dir, args.n_clients, args.random_state)

if __name__ == "__main__":
    main()
