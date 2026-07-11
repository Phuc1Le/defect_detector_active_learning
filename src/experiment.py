import matplotlib.pyplot as plt
import pandas as pd
from active_learning import run_loop
from train import TRANSFORM
import torch
def experiment():
    manifest = pd.read_csv("manifest.csv")
    synthetic = pd.read_csv("synthetic_manifest.csv")
    pool_df = manifest[manifest.split == "pool"]
    test_df = manifest[manifest.split == "test"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    active_hist = run_loop("active", synthetic, pool_df, test_df, device, TRANSFORM)
    random_hist = run_loop("random", synthetic, pool_df, test_df, device, TRANSFORM)

    all_hist = pd.concat([active_hist, random_hist])
    all_hist.to_csv("active_learning_results.csv", index=False)

    plt.figure(figsize=(7,5))
    for strategy, g in all_hist.groupby("strategy"):
        plt.plot(g.labels_used, g.accuracy, marker="o", label=strategy)
    plt.xlabel("Number of real labels used")
    plt.ylabel("Test accuracy")
    plt.title("Active Learning vs Random Labeling — Bottle Defect Detection")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("headline_chart.png", dpi=150)
    plt.show()