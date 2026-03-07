import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from statsmodels.stats.proportion import proportion_confint


BASE = Path(__file__).resolve().parent
PNG_OUT = BASE / "output" / "plots" / "experiment_summary.png"
SVG_OUT = BASE / "output" / "plots" / "experiment_summary.svg"


def load_summary(filename: str) -> dict:
    return json.loads((BASE / filename).read_text())


def build_row(summary_name: str, opponent_key: str, display_name: str, tag: str) -> dict:
    summary = load_summary(summary_name)
    winners = summary["winner_counts"]
    gpt_wins = winners["gpt-4o"]
    opp_wins = winners[opponent_key]
    ties = winners["Tie"]
    decisive = gpt_wins + opp_wins
    ci_low, ci_high = proportion_confint(gpt_wins, decisive, alpha=0.05, method="beta")
    return {
        "title": f"GPT-4o vs {display_name}",
        "tag": tag,
        "gpt_wins": gpt_wins,
        "opp_wins": opp_wins,
        "ties": ties,
        "gpt_pct": gpt_wins / 10,
        "tie_pct": ties / 10,
        "opp_pct": opp_wins / 10,
        "decisive": decisive,
        "gpt_rate": gpt_wins / decisive,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def label_segment(ax, left: float, width: float, y: float, text: str, color: str, inside_min: float = 7.0) -> None:
    if width >= inside_min:
        ax.text(left + width / 2, y, text, ha="center", va="center", fontsize=12, color=color, weight="bold")
    else:
        ax.text(left + width + 1.2, y, text, ha="left", va="center", fontsize=11, color=color, weight="bold")


def draw_badge(ax, x: float, y: float, row: dict, palette: dict) -> None:
    badge = FancyBboxPatch(
        (x, y - 0.21),
        27,
        0.42,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        linewidth=0,
        facecolor=palette["badge"],
        zorder=3,
    )
    ax.add_patch(badge)
    ax.text(x + 13.5, y + 0.05, f"{row['gpt_rate'] * 100:.1f}%", ha="center", va="center", fontsize=21, color="white", weight="bold")
    ax.text(
        x + 13.5,
        y - 0.09,
        f"{row['gpt_wins']} / {row['decisive']} decisive",
        ha="center",
        va="center",
        fontsize=9.5,
        color=palette["badge_text"],
        weight="bold",
    )
    ax.text(
        x + 13.5,
        y - 0.29,
        f"95% CI {row['ci_low'] * 100:.1f}% to {row['ci_high'] * 100:.1f}%",
        ha="center",
        va="top",
        fontsize=9.5,
        color=palette["muted"],
    )


def draw_tag(ax, x: float, y: float, text: str, palette: dict) -> None:
    width = 0.72 * len(text) + 4.5
    tag = FancyBboxPatch(
        (x, y - 0.08),
        width,
        0.16,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=0,
        facecolor="#fff0e4",
        zorder=2,
    )
    ax.add_patch(tag)
    ax.text(x + width / 2, y, text, ha="center", va="center", fontsize=9.5, color=palette["opp"], weight="bold")


def main() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"

    rows = [
        build_row(
            "gpt4o_vs_qwen4b_claude46_summary.json",
            "qwen4b",
            "Qwen 4B",
            "Near parity",
        ),
        build_row(
            "gpt4o_vs_llama31_8b_claude46_summary.json",
            "llama-3.1-8b-instruct",
            "Llama 3.1 8B",
            "Control blowout",
        ),
    ]

    palette = {
        "bg": "#f7f4ed",
        "card": "#fffdf8",
        "card_edge": "#e8e0d2",
        "text": "#16212f",
        "muted": "#66707b",
        "grid": "#ddd4c7",
        "gpt": "#1f6fe5",
        "tie": "#d9dee5",
        "opp": "#ea7b45",
        "badge": "#152844",
        "badge_text": "#cfe0ff",
    }

    fig = plt.figure(figsize=(12.8, 6.7), facecolor=palette["bg"])
    ax = fig.add_axes([0.05, 0.09, 0.9, 0.82])
    ax.set_facecolor(palette["bg"])
    ax.set_xlim(-24, 136)
    ax.set_ylim(-0.65, 1.7)
    ax.axis("off")

    ax.text(0, 1.53, "Experiment Summary", fontsize=12, color=palette["opp"], weight="bold")
    ax.text(0, 1.37, "Pairwise Judge Results on 1,000 Prompts", fontsize=27, color=palette["text"], weight="bold")
    ax.text(0, 1.22, "Claude Opus 4.6 judge, randomized A/B order.", fontsize=11.5, color=palette["muted"])
    ax.text(0, 1.11, "Bars show all prompts; badges show GPT-4o win rate on decisive prompts only.", fontsize=11.5, color=palette["muted"])

    legend_handles = [
        Patch(facecolor=palette["gpt"], edgecolor="none", label="GPT-4o win"),
        Patch(facecolor=palette["tie"], edgecolor="none", label="Tie"),
        Patch(facecolor=palette["opp"], edgecolor="none", label="Opponent win"),
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.69),
        frameon=False,
        ncol=3,
        fontsize=11.5,
        handlelength=1.8,
        columnspacing=1.8,
    )
    for text in legend.get_texts():
        text.set_color(palette["text"])

    ax.text(118, 0.85, "GPT-4o decisive win rate", fontsize=12.5, color=palette["text"], weight="bold", ha="center")

    # Grid and bottom scale for the 0-100% bar region.
    for tick in range(0, 101, 20):
        ax.plot([tick, tick], [-0.28, 0.92], color=palette["grid"], linewidth=1.2, zorder=0)
        ax.text(tick, -0.37, f"{tick}%", ha="center", va="top", fontsize=11, color=palette["muted"])

    for y, row in zip([0.55, -0.02], rows):
        card = FancyBboxPatch(
            (-22, y - 0.25),
            156,
            0.5,
            boxstyle="round,pad=0.02,rounding_size=0.16",
            linewidth=1.1,
            edgecolor=palette["card_edge"],
            facecolor=palette["card"],
            zorder=0.5,
        )
        ax.add_patch(card)

        ax.text(-19.5, y + 0.08, row["title"], ha="left", va="center", fontsize=17.5, color=palette["text"], weight="bold")
        draw_tag(ax, -19.5, y - 0.09, row["tag"], palette)

        bar_y = y - 0.06
        bar_h = 0.16
        ax.add_patch(Rectangle((0, bar_y - bar_h / 2), row["gpt_pct"], bar_h, facecolor=palette["gpt"], edgecolor="none", zorder=2))
        ax.add_patch(Rectangle((row["gpt_pct"], bar_y - bar_h / 2), row["tie_pct"], bar_h, facecolor=palette["tie"], edgecolor="none", zorder=2))
        ax.add_patch(
            Rectangle(
                (row["gpt_pct"] + row["tie_pct"], bar_y - bar_h / 2),
                row["opp_pct"],
                bar_h,
                facecolor=palette["opp"],
                edgecolor="none",
                zorder=2,
            )
        )

        label_segment(ax, 0, row["gpt_pct"], bar_y, f"{row['gpt_wins']}", "white")
        label_segment(ax, row["gpt_pct"], row["tie_pct"], bar_y, f"{row['ties']}", palette["muted"], inside_min=6.5)
        label_segment(ax, row["gpt_pct"] + row["tie_pct"], row["opp_pct"], bar_y, f"{row['opp_wins']}", "white")

        draw_badge(ax, 106, y - 0.01, row, palette)

    ax.text(
        0,
        -0.52,
        "The benchmark clearly separates the Llama control from the Qwen run: GPT-4o is roughly competitive with Qwen, but dominant against Llama 3.1 8B.",
        fontsize=11.2,
        color=palette["text"],
    )

    PNG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_OUT, dpi=220, facecolor=palette["bg"], bbox_inches="tight")
    fig.savefig(SVG_OUT, facecolor=palette["bg"], bbox_inches="tight")
    print(PNG_OUT)
    print(SVG_OUT)


if __name__ == "__main__":
    main()
