import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "gpt4o_vs_qwen4b_claude46_judgments.jsonl"
OUTPUT_PATH = ROOT / "output/plots/qwen_winrate_by_difficulty.png"

KNOWLEDGE_ORDER = ["Low", "Medium", "High"]
REASONING_ORDER = ["Low", "Medium", "High"]


def load_frame(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.open("r", encoding="utf-8")]
    df = pd.DataFrame(rows)
    df["knowledge"] = df["difficulty"].apply(lambda value: value["knowledge"])
    df["reasoning_difficulty"] = df["difficulty"].apply(lambda value: value["reasoning"])
    return df


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_frame(INPUT_PATH)

    records = []
    for knowledge in KNOWLEDGE_ORDER:
        for reasoning in REASONING_ORDER:
            sub = df[
                (df["knowledge"] == knowledge)
                & (df["reasoning_difficulty"] == reasoning)
            ]
            qwen_wins = int((sub["winner"] == "qwen4b").sum())
            gpt_wins = int((sub["winner"] == "gpt-4o").sum())
            ties = int((sub["winner"] == "Tie").sum())
            decisive = qwen_wins + gpt_wins
            winrate = qwen_wins / decisive if decisive else np.nan
            records.append(
                {
                    "knowledge": knowledge,
                    "reasoning": reasoning,
                    "qwen_wins": qwen_wins,
                    "gpt_wins": gpt_wins,
                    "ties": ties,
                    "decisive": decisive,
                    "total": len(sub),
                    "winrate": winrate,
                }
            )

    stats = pd.DataFrame(records)
    heatmap = (
        stats.pivot(index="reasoning", columns="knowledge", values="winrate")
        .reindex(index=REASONING_ORDER, columns=KNOWLEDGE_ORDER)
    )

    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    im = ax.imshow(
        heatmap.to_numpy(),
        cmap="coolwarm",
        vmin=0.35,
        vmax=0.65,
        aspect="auto",
    )

    ax.set_xticks(range(len(KNOWLEDGE_ORDER)))
    ax.set_xticklabels(KNOWLEDGE_ORDER)
    ax.set_yticks(range(len(REASONING_ORDER)))
    ax.set_yticklabels(REASONING_ORDER)
    ax.set_xlabel("Knowledge Difficulty")
    ax.set_ylabel("Reasoning Difficulty")
    ax.set_title(
        "Qwen Win Rate by Prompt Difficulty\n(decisive prompts only; annotations show wins/decisive and ties)",
        loc="left",
        fontweight="bold",
    )

    for row_idx, reasoning in enumerate(REASONING_ORDER):
        for col_idx, knowledge in enumerate(KNOWLEDGE_ORDER):
            row = stats[
                (stats["knowledge"] == knowledge) & (stats["reasoning"] == reasoning)
            ].iloc[0]
            value = row["winrate"]
            label = (
                f"{100 * value:.1f}%\n"
                f"{row['qwen_wins']}/{row['decisive']}\n"
                f"ties={row['ties']}"
            ) if row["decisive"] else f"n={row['total']}\nno decisive"
            text_color = "white" if abs(value - 0.5) > 0.09 else "#111827"
            ax.text(
                col_idx,
                row_idx,
                label,
                ha="center",
                va="center",
                fontsize=10,
                color=text_color,
                fontweight="bold" if row["decisive"] >= 25 else None,
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label("Qwen win rate")
    cbar.ax.yaxis.set_major_formatter(lambda value, _: f"{100 * value:.0f}%")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
