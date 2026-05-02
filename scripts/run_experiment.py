# scripts/run_experiment.py

import pandas as pd

from src.training.trainer import run_cl_experiment
from src.data.task_builder import build_tasks


# -----------------------------
# Config
# -----------------------------

DATASET_MODE = "split_mnist"

METHODS = ["reservoir_uniform", "reservoir_entropy", "reservoir_ig"]

METHOD_LABEL = {
    "no_replay": "No replay",
    "ring_uniform": "Ring + Uniform",
    "reservoir_uniform": "Reservoir + Uniform",
    "reservoir_loss": "Reservoir + Loss-priority",
    "reservoir_entropy": "Reservoir + Entropy-priority",
    "reservoir_ig": "Reservoir + IG-priority",
}

BUDGETS = [10, 20, 50, 100, 200]
SEEDS  = [0, 1, 2, 3, 4]

# Tunables
EPOCHS_PER_TASK = 2
BATCH_SIZE = 128
REPLAY_BS  = 64
REFRESH_EVERY = 200
REFRESH_K = 256
MC_SAMPLES = 10
BETA = 0.9
P_DROP = 0.05
LR = 1e-3


# -----------------------------
# Build tasks
# -----------------------------

train_tasks, test_tasks, NUM_CLASSES, INPUT_SHAPE = build_tasks(DATASET_MODE)


# -----------------------------
# Run experiments
# -----------------------------

rows = []

for B in BUDGETS:
    for m in METHODS:
        for s in SEEDS:

            out = run_cl_experiment(
                method=m,
                buffer_size=B,
                seed=1234 + s,
                train_tasks=train_tasks,
                test_tasks=test_tasks,
                input_shape=INPUT_SHAPE,
                num_classes=NUM_CLASSES,
                epochs_per_task=EPOCHS_PER_TASK,
                batch_size=BATCH_SIZE,
                replay_bs=REPLAY_BS,
                lr=LR,
                p_drop=P_DROP,
                refresh_every=REFRESH_EVERY,
                refresh_k=REFRESH_K,
                mc=MC_SAMPLES,
                beta=BETA
            )

            rows.append({
                "dataset": DATASET_MODE,
                "method": m,
                "method_label": METHOD_LABEL[m],
                "buffer": B,
                "seed": s,
                "final_avg_acc": out["final_avg_acc"],
                "avg_forgetting": out["avg_forgetting"]
            })

            print(f"B={B:3d} | {m:20s} seed={s} acc={out['final_avg_acc']:.3f} forget={out['avg_forgetting']:.3f}")


# -----------------------------
# Save results
# -----------------------------

df = pd.DataFrame(rows)

summary = (
    df.groupby(["method_label", "buffer"])
      .agg(
          acc_mean=("final_avg_acc", "mean"),
          acc_std=("final_avg_acc", "std"),
          forget_mean=("avg_forgetting", "mean"),
          forget_std=("avg_forgetting", "std")
      )
      .reset_index()
)

print(summary.head())

df.to_csv("results/split_mnist/results.csv", index=False)
summary.to_csv("results/split_mnist/summary.csv", index=False)
