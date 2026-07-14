import matplotlib.pyplot as plt
import pandas as pd
from active_learning import run_loop
from train import build_model
from train import TRANSFORM
import copy
import torch
import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def experiment():
    manifest = pd.read_csv("manifest.csv")
    synthetic = pd.read_csv("synthetic_manifest.csv")
    pool_df = manifest[manifest.split == "pool"]
    test_df = manifest[manifest.split == "test"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(0)
    base_model = build_model(device)
    initial_weights = copy.deepcopy(base_model.state_dict())
    all_results = []

    for seed in [0, 1, 2]:
        print(f"\n===== Seed {seed} =====")
        set_seed(seed)
        active_model = build_model(device)
        active_model.load_state_dict(initial_weights)

        random_model = build_model(device)
        random_model.load_state_dict(initial_weights)

        active_hist = run_loop(
            "active",
            synthetic,
            pool_df,
            test_df,
            device,
            TRANSFORM,
            active_model
        )

        random_hist = run_loop(
            "random",
            synthetic,
            pool_df,
            test_df,
            device,
            TRANSFORM,
            random_model
        )

        active_hist["seed"] = seed
        random_hist["seed"] = seed

        all_results.append(active_hist)
        all_results.append(random_hist)

    results = pd.concat(all_results, ignore_index=True)
    summary = (
        results
        .groupby(["strategy", "labels_used"])
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
        )
        .reset_index()
    )
    results.to_csv("active_learning_results.csv", index=False)
    summary.to_csv("active_learning_summary.csv", index=False)
    metrics = [
        ("accuracy", "Test Accuracy", "accuracy_chart.png"),
        ("f1", "Test F1 Score", "f1_chart.png"),
    ]

    for metric, ylabel, filename in metrics:

        plt.figure(figsize=(7,5))

        for strategy in ["active", "random"]:
            d = summary[summary.strategy == strategy]

            plt.plot(
                d.labels_used,
                d[f"{metric}_mean"],
                label=strategy,
            )

            plt.fill_between(
                d.labels_used,
                d[f"{metric}_mean"] - d[f"{metric}_std"],
                d[f"{metric}_mean"] + d[f"{metric}_std"],
                alpha=0.2,
            )

        plt.xlabel("Number of real labels used")
        plt.ylabel(ylabel)
        plt.title(f"Active Learning vs Random Labeling — {ylabel}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(filename, dpi=150)
        plt.show()
    print("\nAverage metrics across seeds:")
    print(summary.round(3))