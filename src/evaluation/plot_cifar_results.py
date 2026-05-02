from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_accuracy_vs_memory(
    summary_csv: str = "results/split_cifar10/accuracy_vs_memory.png",
    output_path: str = "results/split_cifar10/forgetting_vs_memory.png",
) -> None:
    df = pd.read_csv(summary_csv)

    plt.figure(figsize=(8, 5))

    for method_label, group in df.groupby("method_label"):
        group = group.sort_values("buffer")
        plt.plot(group["buffer"], group["acc_mean"], marker="o", label=method_label)

    plt.xlabel("Buffer size")
    plt.ylabel("Final average accuracy")
    plt.title("Accuracy vs Memory Budget")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_forgetting_vs_memory(
    summary_csv: str = "results/summary.csv",
    output_path: str = "results/forgetting_vs_memory.png",
) -> None:
    df = pd.read_csv(summary_csv)

    plt.figure(figsize=(8, 5))

    for method_label, group in df.groupby("method_label"):
        group = group.sort_values("buffer")
        plt.plot(group["buffer"], group["forget_mean"], marker="o", label=method_label)

    plt.xlabel("Buffer size")
    plt.ylabel("Average forgetting")
    plt.title("Forgetting vs Memory Budget")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    Path("results").mkdir(exist_ok=True)

    plot_accuracy_vs_memory()
    plot_forgetting_vs_memory()

    print("Saved plots to results/")


if __name__ == "__main__":
    main()
