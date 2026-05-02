import pandas as pd

from src.training.trainer import run_cl_experiment
from src.data.task_builder import build_tasks
import src.data.task_builder as task_builder

# Force CIFAR experiment
task_builder.DATASET_MODE = "split_cifar10"

#train_tasks, test_tasks, NUM_CLASSES, INPUT_SHAPE = build_tasks()
train_tasks, test_tasks, NUM_CLASSES, INPUT_SHAPE = build_tasks("split_cifar10")

METHODS = [
    "reservoir_uniform",
    "reservoir_entropy",
    "reservoir_ig"
]

METHOD_LABEL = {
    "reservoir_uniform": "Reservoir + Uniform",
    "reservoir_entropy": "Reservoir + Entropy-priority",
    "reservoir_ig": "Reservoir + IG-priority"
}

BUDGETS = [10, 20, 50, 100, 200]
SEEDS = [0, 1, 2, 3, 4]

rows = []

for B in BUDGETS:
    for method in METHODS:
        for seed in SEEDS:

            output = run_cl_experiment(
                method=method,
                buffer_size=B,
                seed=seed,
                train_tasks=train_tasks,
                test_tasks=test_tasks,
                input_shape=INPUT_SHAPE,
                num_classes=NUM_CLASSES,
                epochs_per_task=1,   # CIFAR heavier → keep 1 initially
                batch_size=128,
                replay_bs=64,
                lr=1e-3,
                p_drop=0.1
            )

            rows.append({
                "dataset": "split_cifar10",
                "method": method,
                "method_label": METHOD_LABEL[method],
                "buffer": B,
                "seed": seed,
                "final_avg_acc": output["final_avg_acc"],
                "avg_forgetting": output["avg_forgetting"]
            })

            print(
                f"B={B} | {method} | seed={seed} "
                f"acc={output['final_avg_acc']:.3f} "
                f"forget={output['avg_forgetting']:.3f}"
            )

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

df.to_csv("results/split_cifar10/results.csv", index=False)
summary.to_csv("results/split_cifar10/summary.csv", index=False)

print(summary)
