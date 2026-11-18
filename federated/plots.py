"""
plots.py — Result figures from artifacts/rounds.json + summary.json.

    python -m federated.plots          # writes federated/artifacts/plots/*.png
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .simulate import ARTIFACTS, ROUNDS_PATH, SUMMARY_PATH  # noqa: E402

OUT = ARTIFACTS / "plots"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rounds = json.loads(ROUNDS_PATH.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    # 1. Convergence: accuracy and F1 per round
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for strat, hist in rounds["strategies"].items():
        r = [h["round_number"] for h in hist]
        ax[0].plot(r, [h["metrics"]["accuracy"] for h in hist], marker="o", ms=3, label=strat)
        ax[1].plot(r, [h["metrics"]["f1"] for h in hist], marker="o", ms=3, label=strat)
    if "local_only_mean_accuracy" in summary:
        ax[0].axhline(summary["local_only_mean_accuracy"], ls="--", c="grey", label="local-only mean")
    if "centralized" in summary:
        ax[0].axhline(summary["centralized"]["global_test"]["accuracy"], ls=":", c="black",
                      label="centralized (upper bound)")
    ax[0].set(title="Global test accuracy per round", xlabel="round", ylabel="accuracy")
    ax[1].set(title="Global test F1 per round", xlabel="round", ylabel="F1")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(OUT / "convergence.png", dpi=150)

    # 2. Local-only vs federated vs centralized
    if "local_only" in summary:
        names = list(summary["local_only"]) + [s.upper() for s in summary["federated"]] + ["CENTRAL"]
        vals = ([v["global_test"]["accuracy"] for v in summary["local_only"].values()]
                + [v["final"]["accuracy"] for v in summary["federated"].values()]
                + [summary["centralized"]["global_test"]["accuracy"]])
        colors = (["#9aa5b1"] * len(summary["local_only"]) + ["#0f9d8a"] * len(summary["federated"])
                  + ["#333333"])
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(names, vals, color=colors)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
        ax.set(ylim=(min(vals) - 0.1, 1.0), ylabel="accuracy on campus-wide test set",
               title="Local-only vs federated vs centralized")
        fig.tight_layout()
        fig.savefig(OUT / "local_vs_federated.png", dpi=150)

    # 3. Communication cost
    fig, ax = plt.subplots(figsize=(6, 4))
    for strat, hist in rounds["strategies"].items():
        ax.plot([h["round_number"] for h in hist], [h["cumulative_comm_bytes"] / 1024 for h in hist],
                label=strat)
    ax.set(title="Cumulative communication", xlabel="round", ylabel="KiB")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "comm_cost.png", dpi=150)
    print(f"plots written to {OUT}")


if __name__ == "__main__":
    main()
