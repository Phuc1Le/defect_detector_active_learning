import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def build(cat_dir: Path) -> pd.DataFrame:
    rows = []
    for sub in [cat_dir/"train"/"good", cat_dir/"test"/"good"]:
        for p in sorted(sub.glob("*.png")):
            rows.append({"filepath": str(p), "label": 0})
    test_dir = cat_dir/"test"
    for defect_dir in sorted(test_dir.iterdir()):
        if defect_dir.is_dir() and defect_dir.name != "good":
            for p in sorted(defect_dir.glob("*.png")):
                rows.append({"filepath": str(p), "label": 1})
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category_dir", required=True, help="e.g. data/raw/bottle")
    ap.add_argument("--out", default="manifest.csv")
    ap.add_argument("--test_frac", type=float, default=0.20)
    ap.add_argument("--synth_source_frac", type=float, default=0.40,
                    help="fraction of the *remaining clean* images used to make synthetic defects")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    df = build(Path(args.category_dir))
    print("Total images:", len(df), "| clean:", (df.label == 0).sum(),
          "| defective:", (df.label == 1).sum())
    
    train_pool, test = train_test_split(df, test_size = args.test_frac, stratify=df.label,random_state=args.seed)
    test = test.assign(split="test")

    clean = train_pool[train_pool.label == 0]
    defect = train_pool[train_pool.label == 1]
    synth_source, clean_pool = train_test_split(clean, train_size=args.synth_source_frac,random_state=args.seed)
    synth_source=synth_source.assign(split="synth_source")

    pool = pd.concat([clean_pool, defect]).assign(split="pool")
    out = pd.concat([test, synth_source, pool]).sort_values("filepath")
    out.to_csv(args.out, index=False)

    print("\nSplit sizes:")
    print(out.groupby(["split", "label"]).size())
    print(f"\nWrote {args.out}")

if __name__ == "__main__":
    main()