import os
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / "logs"
PLOTS_DIR = BASE_DIR / "plots"

COST_SCALE = 10**4

RUN_ORDER = ["none", "low", "medium", "jev", "high"]


def _run_label(filename: str) -> str:
    stem = Path(filename).stem
    return stem[len("logs_") :] if stem.startswith("logs_") else stem


def load_logs():
    runs = {}

    for path in sorted(LOGS_DIR.glob("*.csv")):
        df = pd.read_csv(path)

        for column in ("LLM-Latency", "JEV-Latency", "Total Latency", "Total pricing"):
            df[column] = pd.to_numeric(df[column], errors="coerce")

        runs[_run_label(path.name)] = df

    return runs


def _order_labels(labels):
    ordered = [label for label in RUN_ORDER if label in labels]
    ordered += sorted(label for label in labels if label not in RUN_ORDER)

    return ordered


def plot_avg_time(runs):
    labels = _order_labels(runs)
    avg_total_latency = [runs[label]["Total Latency"].mean() for label in labels]

    _, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, avg_total_latency, color="#4C72B0")

    for bar, value in zip(bars, avg_total_latency):
        ax.annotate(
            f"{value:.2f}s",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Run")
    ax.set_ylabel("Average time per question (s)")
    ax.set_title("Average Time per Question")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "avg_time_per_question.png", dpi=150)
    plt.close()


def _normalize(value):
    return str(value).strip().lower().replace(" ", "")


def _answer_accuracy(df):
    correct = 0

    for _, row in df.iterrows():
        if pd.isna(row["LLM-Answer"]):
            continue

        if _normalize(row["answer"]) in _normalize(row["LLM-Answer"]):
            correct += 1

    return correct / len(df)


def _toxicity_accuracy(df):
    compared = 0
    correct = 0

    for _, row in df.iterrows():
        if pd.isna(row["guessed_toxicity"]) or str(row["guessed_toxicity"]).strip() == "":
            continue

        compared += 1

        if _normalize(row["is_toxic"]) == _normalize(row["guessed_toxicity"]):
            correct += 1

    return correct / compared if compared else None


def plot_accuracy(runs):
    labels = _order_labels(runs)
    answer_acc = [100 * _answer_accuracy(runs[label]) for label in labels]
    toxicity_acc = []

    for label in labels:
        acc = _toxicity_accuracy(runs[label])
        toxicity_acc.append(100 * acc if acc is not None else None)

    x = range(len(labels))
    width = 0.35

    _, ax = plt.subplots(figsize=(8, 5))
    bars_answer = ax.bar(
        [xi - width / 2 for xi in x],
        answer_acc,
        width,
        label="Answer accuracy",
        color="#4C72B0",
    )

    bars_toxicity = []

    for xi, acc in zip(x, toxicity_acc):
        if acc is not None:
            bars_toxicity.append(
                ax.bar(xi + width / 2, acc, width, label="Toxicity accuracy", color="#DD8452")[0]
            )

    for bar, value in zip(bars_answer, answer_acc):
        ax.annotate(
            f"{value:.0f}%",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for bar, value in zip(bars_toxicity, [a for a in toxicity_acc if a is not None]):
        ax.annotate(
            f"{value:.0f}%",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Run")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy of Toxicity Detection and Answers")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "accuracy.png", dpi=150)
    plt.close()


def plot_cost(runs):
    labels = _order_labels(runs)
    avg_cost = [runs[label]["Total pricing"].mean() * COST_SCALE for label in labels]

    _, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, avg_cost, color="#4C72B0")

    for bar, value in zip(bars, avg_cost):
        ax.annotate(
            f"{value:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Run")
    ax.set_ylabel(f"Average cost per question ($ \\times 10^{{-4}}$)")
    ax.set_title("Average Cost per Question")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cost_per_question.png", dpi=150)
    plt.close()


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    runs = load_logs()

    if not runs:
        raise FileNotFoundError(f"No CSV logs found in {LOGS_DIR}")

    plot_avg_time(runs)
    plot_accuracy(runs)
    plot_cost(runs)

    print(f"Plots written to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
