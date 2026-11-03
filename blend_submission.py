"""Blend two submission files in log1p space.

usage: python blend_submission.py TF.csv LGBM.csv -w 0.3 -o OUT.csv
"""
import argparse

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tf", help="TimesFM submission csv")
    ap.add_argument("lgbm", help="LightGBM submission csv")
    ap.add_argument("-w", type=float, default=0.3, help="weight on the LGBM file (default 0.3)")
    ap.add_argument("-o", default="outputs/sub_blend.csv", help="output csv")
    args = ap.parse_args()

    tf = pd.read_csv(args.tf)
    lg = pd.read_csv(args.lgbm)
    assert (tf["id"].to_numpy() == lg["id"].to_numpy()).all(), "id columns differ"

    sales = np.expm1((1 - args.w) * np.log1p(tf["sales"].to_numpy()) + args.w * np.log1p(lg["sales"].to_numpy()))
    out = pd.DataFrame({"id": tf["id"], "sales": np.maximum(sales, 0)})
    out.to_csv(args.o, index=False)
    print(f"w={args.w}, rows={len(out)}, mean={out['sales'].mean():.1f} -> {args.o}")


if __name__ == "__main__":
    main()
