import pandas as pd, torch, copy
import random
import numpy as np
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
def pick_batch_active(unrevealed_df, probs, k):
    tmp = unrevealed_df.copy()
    tmp["prob"] = probs
    tmp["uncertainty"] = (tmp["prob"] - 0.5).abs()     # 0 = maximally uncertain
    return tmp.nsmallest(k, "uncertainty").drop(columns=["prob", "uncertainty"])

def pick_batch_random(unrevealed_df, probs, k, seed):
    return unrevealed_df.sample(n=min(k, len(unrevealed_df)), random_state=seed)

def run_loop(strategy, synthetic_df, pool_df, test_df, device, transform, model,
             k=20, n_rounds=5, seed=0):
    from train import train_model
    from evaluate import get_probs_and_preds, report

    training_set = synthetic_df.copy()
    unrevealed = pool_df.copy()
    history = []
    for round_i in range(n_rounds):
        print(training_set.label.value_counts())
        model = train_model(model, training_set, device)
        metrics = report(model, test_df, device, transform)
        history.append({"round": round_i, "labels_used": len(training_set) - len(synthetic_df),
                         "strategy": strategy, **metrics})
        print(f"[{strategy}] round {round_i}: labels_used={history[-1]['labels_used']} "
              f"acc={metrics['accuracy']:.3f}")

        if len(unrevealed) == 0:
            break

        probs, _, _ = get_probs_and_preds(model, unrevealed, device, transform)
        if strategy == "active":
            chosen = pick_batch_active(unrevealed, probs, k)
        elif strategy == "random":
            chosen = pick_batch_random(unrevealed, probs, k, seed=seed + round_i)
        else:
            raise ValueError(strategy)

        training_set = pd.concat([training_set, chosen])
        unrevealed = unrevealed.drop(chosen.index)

    return pd.DataFrame(history)