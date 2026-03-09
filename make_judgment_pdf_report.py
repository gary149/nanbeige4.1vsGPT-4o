import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from scipy.stats import beta, binomtest, chi2_contingency, fisher_exact


REPORT_TITLE = "GPT-4o vs Nanbeige 4.1-3B Judge Report"
MODEL_B_LABEL = "nanbeige4.1-3b"
MODEL_B_DISPLAY = "Nanbeige 4.1-3B"
OUTCOME_ORDER = [MODEL_B_LABEL, "gpt-4o", "Tie"]
OUTCOME_LABELS = {
    MODEL_B_LABEL: f"{MODEL_B_DISPLAY} wins",
    "gpt-4o": "GPT-4o wins",
    "Tie": "Ties",
}
OUTCOME_COLORS = {
    MODEL_B_LABEL: "#C4493A",
    "gpt-4o": "#2457A5",
    "Tie": "#8F98A3",
}
DISPLAY_LABELS = {
    MODEL_B_LABEL: MODEL_B_DISPLAY,
    "gpt-4o": "GPT-4o",
    "Tie": "Tie",
}
REASON_PATTERNS = {
    "Detail and completeness": r"\b(?:detail(?:ed)?|complete(?:ness)?|thorough|granular|covers?|addresses?)\b",
    "Accuracy and factuality": r"\b(?:accur(?:ate|acy)|fact(?:ual)?|technically correct|correct(?:ness)?)\b",
    "Structure and coherence": r"\b(?:coheren|logical|structure[ds]?|organized?|flow|consistent)\b",
    "Prose and naturalness": r"\b(?:prose|natural|readab|well-written|awkward|stilted|fluent)\b",
    "Creativity and originality": r"\b(?:creative|original|imaginative|voice)\b",
    "Concision and directness": r"\b(?:concise|brief|direct|succinct)\b",
    "Errors in losing answer": r"\b(?:inaccur|incorrect|unsupported|fabricat|hallucin|contradict|misattrib)\b",
}


@dataclass
class SummaryStat:
    wins: int
    losses: int
    n: int
    share: float
    p_value: float
    ci_low: float
    ci_high: float


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Create a PDF report for the judgment run.")
    parser.add_argument(
        "--judgments",
        type=Path,
        default=repo_root / "gpt4o_vs_nanbeige_claude46_judgments.jsonl",
        help="Path to the full judgment JSONL file.",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=repo_root / "output/pdf/gpt4o_vs_nanbeige_report.pdf",
        help="Destination PDF path.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=repo_root / "output/pdf/gpt4o_vs_nanbeige_report_summary.json",
        help="Destination summary JSON path.",
    )
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=repo_root / "tmp/pdfs/gpt4o_vs_nanbeige_report_assets",
        help="Directory for intermediate chart images.",
    )
    return parser.parse_args()


def exact_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    if k == 0:
        lower = 0.0
    else:
        lower = beta.ppf(alpha / 2, k, n - k + 1)
    if k == n:
        upper = 1.0
    else:
        upper = beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lower), float(upper))


def exact_summary(wins: int, losses: int) -> SummaryStat:
    decisive_n = wins + losses
    share = wins / decisive_n if decisive_n else float("nan")
    p_value = binomtest(wins, decisive_n, p=0.5, alternative="two-sided").pvalue if decisive_n else float("nan")
    ci_low, ci_high = exact_ci(wins, decisive_n) if decisive_n else (float("nan"), float("nan"))
    return SummaryStat(
        wins=wins,
        losses=losses,
        n=decisive_n,
        share=share,
        p_value=float(p_value),
        ci_low=ci_low,
        ci_high=ci_high,
    )


def benjamini_hochberg(p_values: Iterable[float]) -> list[float]:
    p_values = list(p_values)
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running_min = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        value = p_values[idx]
        adjusted_value = min(running_min, value * m / rank)
        adjusted[idx] = adjusted_value
        running_min = adjusted_value
    return adjusted.tolist()


def load_data(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.open("r", encoding="utf-8")]
    df = pd.DataFrame(rows)
    df["knowledge"] = df["difficulty"].apply(lambda value: value["knowledge"])
    df["reasoning_difficulty"] = df["difficulty"].apply(lambda value: value["reasoning"])
    df["decisive"] = df["winner"] != "Tie"
    df["qwen_win"] = df["winner"] == MODEL_B_LABEL
    df["gpt_win"] = df["winner"] == "gpt-4o"
    df["a_win"] = df["judgement"] == "A"
    df["b_win"] = df["judgement"] == "B"
    df["qwen_is_a"] = df["response_a_model"] == MODEL_B_LABEL
    df["prompt_chars"] = df["prompt"].str.len()
    df["gpt_chars"] = df["gpt-4o"].str.len()
    df["qwen_chars"] = df[MODEL_B_LABEL].str.len()
    df["qwen_minus_gpt_chars"] = df["qwen_chars"] - df["gpt_chars"]
    df["qwen_longer"] = df["qwen_chars"] > df["gpt_chars"]
    df["prompt_length_bucket"] = pd.qcut(
        df["prompt_chars"],
        4,
        labels=["Q1 shortest", "Q2", "Q3", "Q4 longest"],
        duplicates="drop",
    )
    return df


def subgroup_table(df: pd.DataFrame, column: str, min_total_n: int = 1) -> pd.DataFrame:
    records = []
    for group, group_df in df.groupby(column, dropna=False, observed=False):
        qwen_wins = int((group_df["winner"] == MODEL_B_LABEL).sum())
        gpt_wins = int((group_df["winner"] == "gpt-4o").sum())
        ties = int((group_df["winner"] == "Tie").sum())
        total_n = len(group_df)
        if total_n < min_total_n or qwen_wins + gpt_wins == 0:
            continue
        stat = exact_summary(qwen_wins, gpt_wins)
        records.append(
            {
                "group": str(group),
                "qwen_wins": qwen_wins,
                "gpt_wins": gpt_wins,
                "ties": ties,
                "total_n": total_n,
                "decisive_n": stat.n,
                "qwen_share": stat.share,
                "ci_low": stat.ci_low,
                "ci_high": stat.ci_high,
                "p_value": stat.p_value,
            }
        )
    table = pd.DataFrame(records)
    if table.empty:
        return table
    table = table.sort_values(["qwen_share", "decisive_n"], ascending=[False, False]).reset_index(drop=True)
    table["p_adj"] = benjamini_hochberg(table["p_value"].tolist())
    table["significant"] = table["p_adj"] < 0.05
    table["leader"] = np.where(table["qwen_share"] > 0.5, MODEL_B_LABEL, "gpt-4o")
    return table


def build_position_stats(df: pd.DataFrame) -> dict:
    decisive = df[df["decisive"]]
    a_vs_b = exact_summary(int(decisive["a_win"].sum()), int(decisive["b_win"].sum()))

    qwen_as_a = decisive[decisive["qwen_is_a"]]
    qwen_as_b = decisive[~decisive["qwen_is_a"]]
    qwen_a_summary = exact_summary(
        int((qwen_as_a["winner"] == MODEL_B_LABEL).sum()),
        int((qwen_as_a["winner"] == "gpt-4o").sum()),
    )
    qwen_b_summary = exact_summary(
        int((qwen_as_b["winner"] == MODEL_B_LABEL).sum()),
        int((qwen_as_b["winner"] == "gpt-4o").sum()),
    )
    contingency = np.array(
        [
            [qwen_a_summary.wins, qwen_a_summary.losses],
            [qwen_b_summary.wins, qwen_b_summary.losses],
        ]
    )
    odds_ratio, fisher_p = fisher_exact(contingency, alternative="two-sided")
    return {
        "a_vs_b": {
            "a_wins": a_vs_b.wins,
            "b_wins": a_vs_b.losses,
            "share_a": a_vs_b.share,
            "ci_low": a_vs_b.ci_low,
            "ci_high": a_vs_b.ci_high,
            "p_value": a_vs_b.p_value,
        },
        "qwen_as_a": {
            "wins": qwen_a_summary.wins,
            "losses": qwen_a_summary.losses,
            "share": qwen_a_summary.share,
            "ci_low": qwen_a_summary.ci_low,
            "ci_high": qwen_a_summary.ci_high,
        },
        "qwen_as_b": {
            "wins": qwen_b_summary.wins,
            "losses": qwen_b_summary.losses,
            "share": qwen_b_summary.share,
            "ci_low": qwen_b_summary.ci_low,
            "ci_high": qwen_b_summary.ci_high,
        },
        "qwen_position_fisher_p": float(fisher_p),
        "qwen_position_odds_ratio": float(odds_ratio),
    }


def build_length_stats(df: pd.DataFrame) -> dict:
    decisive = df[df["decisive"]]
    qwen_longer = decisive[decisive["qwen_longer"]]
    gpt_longer_or_equal = decisive[~decisive["qwen_longer"]]

    qwen_longer_summary = exact_summary(
        int((qwen_longer["winner"] == MODEL_B_LABEL).sum()),
        int((qwen_longer["winner"] == "gpt-4o").sum()),
    )
    gpt_longer_summary = exact_summary(
        int((gpt_longer_or_equal["winner"] == MODEL_B_LABEL).sum()),
        int((gpt_longer_or_equal["winner"] == "gpt-4o").sum()),
    )
    contingency = np.array(
        [
            [qwen_longer_summary.wins, qwen_longer_summary.losses],
            [gpt_longer_summary.wins, gpt_longer_summary.losses],
        ]
    )
    odds_ratio, fisher_p = fisher_exact(contingency, alternative="two-sided")
    return {
        "qwen_longer_on_all_prompts": float(df["qwen_longer"].mean()),
        "avg_qwen_minus_gpt_chars": float(df["qwen_minus_gpt_chars"].mean()),
        "median_qwen_minus_gpt_chars": float(df["qwen_minus_gpt_chars"].median()),
        "qwen_longer": {
            "decisive_n": qwen_longer_summary.n,
            "share": qwen_longer_summary.share,
            "ci_low": qwen_longer_summary.ci_low,
            "ci_high": qwen_longer_summary.ci_high,
        },
        "gpt_longer_or_equal": {
            "decisive_n": gpt_longer_summary.n,
            "share": gpt_longer_summary.share,
            "ci_low": gpt_longer_summary.ci_low,
            "ci_high": gpt_longer_summary.ci_high,
        },
        "fisher_p": float(fisher_p),
        "odds_ratio": float(odds_ratio),
    }


def build_cost_stats(df: pd.DataFrame) -> dict:
    return {
        "total_cost": float(df["cost"].sum()),
        "mean_cost": float(df["cost"].mean()),
        "median_cost": float(df["cost"].median()),
        "p10_cost": float(df["cost"].quantile(0.10)),
        "p90_cost": float(df["cost"].quantile(0.90)),
        "max_cost": float(df["cost"].max()),
        "min_cost": float(df["cost"].min()),
    }


def build_reason_pattern_table(df: pd.DataFrame) -> pd.DataFrame:
    decisive = df[df["decisive"]].copy()
    decisive["judge_reasoning_lower"] = decisive["judge_reasoning"].str.lower().fillna("")
    records = []
    for label, pattern in REASON_PATTERNS.items():
        qwen_rate = decisive.loc[decisive["winner"] == MODEL_B_LABEL, "judge_reasoning_lower"].str.contains(pattern, regex=True).mean()
        gpt_rate = decisive.loc[decisive["winner"] == "gpt-4o", "judge_reasoning_lower"].str.contains(pattern, regex=True).mean()
        records.append(
            {
                "pattern": label,
                "qwen_rate": float(qwen_rate),
                "gpt_rate": float(gpt_rate),
                "delta_qwen_minus_gpt": float(qwen_rate - gpt_rate),
            }
        )
    table = pd.DataFrame(records).sort_values("delta_qwen_minus_gpt", ascending=False).reset_index(drop=True)
    return table


def format_pct(value: float) -> str:
    if math.isnan(value):
        return "N/A"
    return f"{100 * value:.1f}%"


def format_p_value(value: float) -> str:
    if value < 0.001:
        return "< 0.001"
    return f"{value:.3f}"


def plot_overall_outcomes(df: pd.DataFrame, output_path: Path) -> None:
    counts = [int((df["winner"] == outcome).sum()) for outcome in OUTCOME_ORDER]
    labels = [OUTCOME_LABELS[outcome] for outcome in OUTCOME_ORDER]
    colors_list = [OUTCOME_COLORS[outcome] for outcome in OUTCOME_ORDER]

    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.barh(labels, counts, color=colors_list, edgecolor="white", linewidth=1.5)
    ax.set_xlabel("Prompt count")
    ax.set_title(f"Overall head-to-head result (n = {sum(counts)})", loc="left", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max(counts) * 1.18)
    for idx, count in enumerate(counts):
        ax.text(count + 10, idx, str(count), va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_forest(table: pd.DataFrame, title: str, output_path: Path, max_rows: int = 12) -> None:
    plot_df = table.head(max_rows).copy()
    plot_df = plot_df.sort_values("qwen_share", ascending=True)
    y_positions = np.arange(len(plot_df))

    fig_height = max(3.0, 0.48 * len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(8.2, fig_height))

    colors_used = [
        OUTCOME_COLORS[MODEL_B_LABEL] if leader == MODEL_B_LABEL else OUTCOME_COLORS["gpt-4o"]
        for leader in plot_df["leader"]
    ]
    x = plot_df["qwen_share"].to_numpy()
    lower = x - plot_df["ci_low"].to_numpy()
    upper = plot_df["ci_high"].to_numpy() - x

    ax.errorbar(
        x,
        y_positions,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color="#20252B",
        ecolor="#7C8794",
        elinewidth=1.4,
        capsize=3,
        markersize=5,
        zorder=3,
    )
    ax.scatter(x, y_positions, s=48, color=colors_used, zorder=4)
    ax.axvline(0.5, color="#6B7280", linestyle="--", linewidth=1)
    ax.set_xlim(0.15, 0.85)
    ax.set_yticks(y_positions)
    labels = []
    for _, row in plot_df.iterrows():
        suffix = " *" if row["significant"] else ""
        labels.append(f"{row['group']} (n={int(row['decisive_n'])}){suffix}")
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"{MODEL_B_DISPLAY} share among decisive prompts")
    ax.xaxis.set_major_formatter(lambda value, _: f"{100 * value:.0f}%")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_two_group_share(stats: dict, output_path: Path) -> None:
    all_labels = [f"{MODEL_B_DISPLAY} longer", "GPT longer or equal"]
    all_shares = [stats["qwen_longer"]["share"], stats["gpt_longer_or_equal"]["share"]]
    all_ci_low = [stats["qwen_longer"]["ci_low"], stats["gpt_longer_or_equal"]["ci_low"]]
    all_ci_high = [stats["qwen_longer"]["ci_high"], stats["gpt_longer_or_equal"]["ci_high"]]
    all_ns = [stats["qwen_longer"]["decisive_n"], stats["gpt_longer_or_equal"]["decisive_n"]]
    all_colors = [OUTCOME_COLORS[MODEL_B_LABEL], OUTCOME_COLORS["gpt-4o"]]

    # Filter out groups with 0 decisive prompts (share is NaN)
    mask = [n > 0 for n in all_ns]
    labels = [v for v, m in zip(all_labels, mask) if m]
    shares = [v for v, m in zip(all_shares, mask) if m]
    ci_low = [v for v, m in zip(all_ci_low, mask) if m]
    ci_high = [v for v, m in zip(all_ci_high, mask) if m]
    ns = [v for v, m in zip(all_ns, mask) if m]
    colors_used = [v for v, m in zip(all_colors, mask) if m]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x_positions = np.arange(len(labels))
    ax.bar(x_positions, shares, color=colors_used, width=0.58)
    ax.errorbar(
        x_positions,
        shares,
        yerr=[
            [share - low for share, low in zip(shares, ci_low)],
            [high - share for share, high in zip(shares, ci_high)],
        ],
        fmt="none",
        ecolor="#20252B",
        capsize=4,
        linewidth=1.4,
    )
    ax.axhline(0.5, color="#6B7280", linestyle="--", linewidth=1)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel(f"{MODEL_B_DISPLAY} share among decisive prompts")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{label}\n(n={n})" for label, n in zip(labels, ns)])
    ax.yaxis.set_major_formatter(lambda value, _: f"{100 * value:.0f}%")
    ax.set_title("Length advantage and win rate", loc="left", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_styles():
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=28,
            textColor=colors.HexColor("#152238"),
            spaceAfter=12,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#2457A5"),
            spaceBefore=8,
            spaceAfter=8,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.6,
            leading=13.5,
            textColor=colors.HexColor("#20252B"),
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.4,
            leading=11,
            textColor=colors.HexColor("#4B5563"),
            spaceAfter=4,
        ),
        "table": ParagraphStyle(
            "Table",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=10.5,
            textColor=colors.HexColor("#20252B"),
        ),
        "table_bold": ParagraphStyle(
            "TableBold",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=10.5,
            textColor=colors.HexColor("#152238"),
        ),
    }
    return styles


def add_table(table_data: list[list[str]], col_widths: list[float]) -> Table:
    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF9")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#152238")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LEADING", (0, 0), (-1, -1), 10.5),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D2D8E0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def make_cover_story(styles, analysis: dict) -> list:
    overall = analysis["overall"]
    cost_stats = analysis["cost_stats"]
    summary_points = [
        f"{MODEL_B_DISPLAY} won {overall['qwen_wins']} prompts, GPT-4o won {overall['gpt_wins']}, and {overall['ties']} ended in ties.",
        f"On decisive prompts only, {MODEL_B_DISPLAY} won {format_pct(overall['qwen_share'])} of cases "
        f"(95% CI {format_pct(overall['qwen_ci_low'])} to {format_pct(overall['qwen_ci_high'])}, "
        f"exact binomial p = {format_p_value(overall['qwen_p_value'])}).",
        f"The full {overall['total_n']}-prompt judging run cost {cost_stats['total_cost']:.2f} OpenRouter credits, "
        f"or {cost_stats['mean_cost']:.3f} credits per prompt on average.",
    ]
    story = [
        Spacer(1, 0.3 * inch),
        Paragraph(REPORT_TITLE, styles["title"]),
        Paragraph(
            f"A statistical review of the {overall['total_n']}-prompt pairwise evaluation judged by Claude Opus 4.6.",
            styles["body"],
        ),
        Spacer(1, 0.1 * inch),
    ]
    for point in summary_points:
        story.append(Paragraph(f"- {point}", styles["body"]))
    story.append(Spacer(1, 0.25 * inch))
    story.append(
        Paragraph(
            "Interpretation standard: exact binomial tests are used for paired win/loss comparisons after excluding ties. "
            "Subgroup families use Benjamini-Hochberg correction, and small-sample language buckets are treated as exploratory.",
            styles["small"],
        )
    )
    return story


def build_pdf(
    pdf_path: Path,
    asset_paths: dict[str, Path],
    analysis: dict,
    genre_table: pd.DataFrame,
    language_table: pd.DataFrame,
    knowledge_table: pd.DataFrame,
    reasoning_table: pd.DataFrame,
    prompt_length_table: pd.DataFrame,
    reason_patterns: pd.DataFrame,
) -> None:
    styles = build_styles()
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        topMargin=0.6 * inch,
        bottomMargin=0.55 * inch,
        leftMargin=0.65 * inch,
        rightMargin=0.65 * inch,
        title=REPORT_TITLE,
        author="Codex",
    )

    story = []
    story.extend(make_cover_story(styles, analysis))
    story.append(Image(str(asset_paths["overall"]), width=7.0 * inch, height=3.05 * inch))
    story.append(Spacer(1, 0.18 * inch))

    overall = analysis["overall"]
    position = analysis["position_stats"]
    methodology = [
        f"{overall['total_n']} prompts were sampled from train.jsonl with unique first-user prompts.",
        f"Each prompt received one answer from GPT-4o and one answer from {MODEL_B_DISPLAY}.",
        "Claude Opus 4.6 judged the pair with randomized A/B order, visible reasoning, and max verbosity through OpenRouter.",
        "Overall and subgroup significance tests use exact binomial tests on decisive prompts only. Ties are reported separately.",
        "Within each subgroup family, p-values are Benjamini-Hochberg adjusted to reduce false discoveries.",
    ]

    story.append(Paragraph("Methodology", styles["subtitle"]))
    for item in methodology:
        story.append(Paragraph(f"- {item}", styles["body"]))

    story.append(Paragraph("Executive summary", styles["subtitle"]))
    summary_table = add_table(
        [
            ["Metric", "Value"],
            ["Total prompts", f"{overall['total_n']}"],
            [f"{MODEL_B_DISPLAY} wins", f"{overall['qwen_wins']}"],
            ["GPT-4o wins", f"{overall['gpt_wins']}"],
            ["Ties", f"{overall['ties']}"],
            [f"{MODEL_B_DISPLAY} share on decisive prompts", f"{format_pct(overall['qwen_share'])}"],
            ["95% CI", f"{format_pct(overall['qwen_ci_low'])} to {format_pct(overall['qwen_ci_high'])}"],
            ["Exact binomial p-value", format_p_value(overall["qwen_p_value"])],
            ["Judge cost", f"{analysis['cost_stats']['total_cost']:.2f} credits"],
        ],
        [2.7 * inch, 3.2 * inch],
    )
    story.append(summary_table)
    story.append(Spacer(1, 0.14 * inch))
    story.append(
        Paragraph(
            f"Result: {MODEL_B_DISPLAY} leads by {overall['qwen_wins'] - overall['gpt_wins']} wins on decisive prompts, "
            f"or {100 * (overall['qwen_share'] - 0.5):.1f} percentage points over a 50/50 split.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            f"Order-bias check: A was chosen on {format_pct(position['a_vs_b']['share_a'])} of decisive prompts "
            f"(p = {format_p_value(position['a_vs_b']['p_value'])}), and {MODEL_B_DISPLAY}'s win rate was "
            f"{format_pct(position['qwen_as_a']['share'])} when shown as A vs "
            f"{format_pct(position['qwen_as_b']['share'])} when shown as B "
            f"(Fisher p = {format_p_value(position['qwen_position_fisher_p'])}).",
            styles["body"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("Where each model is stronger", styles["subtitle"]))
    story.append(
        Paragraph(
            f"The plots below show {MODEL_B_DISPLAY}'s share among decisive prompts. The dashed line marks parity at 50%. "
            "Rows with an asterisk remain significant after Benjamini-Hochberg correction within that subgroup family.",
            styles["body"],
        )
    )
    story.append(Image(str(asset_paths["genre"]), width=7.0 * inch, height=4.5 * inch))
    story.append(Spacer(1, 0.08 * inch))
    story.append(Image(str(asset_paths["language"]), width=7.0 * inch, height=4.85 * inch))

    significant_qwen_genres = genre_table[(genre_table["significant"]) & (genre_table["leader"] == MODEL_B_LABEL)]
    significant_gpt_genres = genre_table[(genre_table["significant"]) & (genre_table["leader"] == "gpt-4o")]
    major_language_hits = language_table[(language_table["significant"]) & (language_table["decisive_n"] >= 20)]

    if not significant_qwen_genres.empty:
        leaders = ", ".join(
            f"{row.group} ({format_pct(row.qwen_share)} {MODEL_B_DISPLAY} share)"
            for row in significant_qwen_genres.itertuples()
        )
        story.append(Paragraph(f"Significant genre pockets favoring {MODEL_B_DISPLAY}: {leaders}.", styles["body"]))
    if not significant_gpt_genres.empty:
        leaders = ", ".join(
            f"{row.group} ({format_pct(1 - row.qwen_share)} GPT-4o share)"
            for row in significant_gpt_genres.itertuples()
        )
        story.append(Paragraph(f"Significant genre pockets favoring GPT-4o: {leaders}.", styles["body"]))
    else:
        story.append(
            Paragraph(
                "No genre bucket shows a statistically significant advantage for GPT-4o after multiple-testing correction.",
                styles["body"],
            )
        )

    if not major_language_hits.empty:
        language_lines = []
        for row in major_language_hits.itertuples():
            favored = MODEL_B_DISPLAY if row.leader == MODEL_B_LABEL else "GPT-4o"
            favored_share = row.qwen_share if row.leader == MODEL_B_LABEL else 1 - row.qwen_share
            language_lines.append(f"{row.group}: {favored} at {format_pct(favored_share)} of decisive prompts")
        story.append(Paragraph("Significant language pockets: " + "; ".join(language_lines) + ".", styles["body"]))

    story.append(PageBreak())
    story.append(Paragraph("Difficulty, prompt shape, and style effects", styles["subtitle"]))
    story.append(Image(str(asset_paths["difficulty"]), width=7.0 * inch, height=4.75 * inch))
    story.append(Spacer(1, 0.08 * inch))
    story.append(Image(str(asset_paths["length"]), width=6.3 * inch, height=3.0 * inch))

    length_stats = analysis["length_stats"]
    length_text = f"{MODEL_B_DISPLAY} answers are longer on {format_pct(length_stats['qwen_longer_on_all_prompts'])} of all prompts. "
    length_text += f"When {MODEL_B_DISPLAY} is longer, it wins {format_pct(length_stats['qwen_longer']['share'])} of decisive prompts"
    if length_stats["gpt_longer_or_equal"]["decisive_n"] > 0:
        length_text += (
            f"; when GPT-4o is longer or equal, {MODEL_B_DISPLAY} still wins "
            f"{format_pct(length_stats['gpt_longer_or_equal']['share'])}. "
            f"The difference is statistically significant by Fisher's exact test (p = {format_p_value(length_stats['fisher_p'])})."
        )
    else:
        length_text += f". GPT-4o was never longer or equal in this sample, so the length-bias comparison is not applicable."
    story.append(Paragraph(length_text, styles["body"]))
    story.append(
        Paragraph(
            f"Average answer length difference is {length_stats['avg_qwen_minus_gpt_chars']:.0f} characters in {MODEL_B_DISPLAY}'s favor "
            f"(median {length_stats['median_qwen_minus_gpt_chars']:.0f}). That points to a meaningful completeness advantage, "
            f"especially on open-ended prompts, but it is not the whole story because {MODEL_B_DISPLAY} still leads even when it is not longer.",
            styles["body"],
        )
    )

    story.append(Paragraph("Difficulty table", styles["subtitle"]))
    tb = styles["table_bold"]
    difficulty_rows = [[
        Paragraph("Bucket", tb),
        Paragraph(f"{MODEL_B_DISPLAY} wins", tb),
        Paragraph("GPT-4o wins", tb),
        Paragraph("Ties", tb),
        Paragraph(f"{MODEL_B_DISPLAY} share", tb),
        Paragraph("Adj. p", tb),
    ]]
    for label, table in [("Knowledge", knowledge_table), ("Reasoning", reasoning_table), ("Prompt length", prompt_length_table)]:
        for row in table.itertuples():
            difficulty_rows.append(
                [
                    f"{label}: {row.group}",
                    str(row.qwen_wins),
                    str(row.gpt_wins),
                    str(row.ties),
                    format_pct(row.qwen_share),
                    format_p_value(row.p_adj),
                ]
            )
    story.append(add_table(difficulty_rows, [1.9 * inch, 1.0 * inch, 0.85 * inch, 0.55 * inch, 1.0 * inch, 0.75 * inch]))

    story.append(PageBreak())
    story.append(Paragraph("Qualitative read on the judge rationales", styles["subtitle"]))
    story.append(
        Paragraph(
            "The judge rationales were not used as a second scoring model, but they do reveal which rubric dimensions kept recurring in wins. "
            "The table below shows the share of winning rationales that mention each theme.",
            styles["body"],
        )
    )
    reason_rows = [[
        Paragraph("Theme", tb),
        Paragraph(f"{MODEL_B_DISPLAY} win rate", tb),
        Paragraph("GPT-4o win rate", tb),
        Paragraph("Delta", tb),
    ]]
    for row in reason_patterns.itertuples():
        reason_rows.append(
            [
                row.pattern,
                format_pct(row.qwen_rate),
                format_pct(row.gpt_rate),
                f"{100 * row.delta_qwen_minus_gpt:+.1f} pp",
            ]
        )
    story.append(add_table(reason_rows, [2.25 * inch, 1.1 * inch, 1.05 * inch, 0.9 * inch]))

    qwen_leading_patterns = reason_patterns.head(3)
    gpt_leading_patterns = reason_patterns.tail(3).iloc[::-1]
    story.append(Spacer(1, 0.1 * inch))
    story.append(
        Paragraph(
            f"Common {MODEL_B_DISPLAY} win themes: "
            + "; ".join(
                f"{row.pattern} ({format_pct(row.qwen_rate)} vs {format_pct(row.gpt_rate)})"
                for row in qwen_leading_patterns.itertuples()
            )
            + ".",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "Common GPT-4o win themes: "
            + "; ".join(
                f"{row.pattern} ({format_pct(row.gpt_rate)} vs {format_pct(row.qwen_rate)})"
                for row in gpt_leading_patterns.itertuples()
            )
            + ".",
            styles["body"],
        )
    )

    story.append(Paragraph("Limitations", styles["subtitle"]))
    limitations = [
        "This benchmark uses one sampled answer per model per prompt, so it measures one-shot quality rather than best-of-n capability.",
        "All judgments come from one strong judge model. That is useful for consistency, but it is still a single-judge design.",
        "The language analysis is only stable for large language buckets. Rare languages are reported descriptively but should not drive major conclusions.",
        "Pairwise verdicts do not expose margin sizes, so the report focuses on win rates, confidence intervals, and repeated patterns rather than a hidden score scale.",
    ]
    for item in limitations:
        story.append(Paragraph(f"- {item}", styles["body"]))

    story.append(Spacer(1, 0.08 * inch))

    # Build bottom-line paragraph dynamically from subgroup results
    bottom_parts = [
        f"Bottom line: {MODEL_B_DISPLAY} wins the benchmark overall "
        f"({overall['qwen_wins']} to {overall['gpt_wins']} with {overall['ties']} ties)."
    ]
    sig_qwen = genre_table[(genre_table["significant"]) & (genre_table["leader"] == MODEL_B_LABEL)]
    sig_gpt = genre_table[(genre_table["significant"]) & (genre_table["leader"] == "gpt-4o")]
    sig_lang = language_table[(language_table["significant"]) & (language_table["decisive_n"] >= 5)] if not language_table.empty else language_table
    if not sig_qwen.empty:
        pockets = ", ".join(sig_qwen["group"].tolist())
        bottom_parts.append(f" The strongest evidence for {MODEL_B_DISPLAY} concentrates in {pockets}.")
    if not sig_gpt.empty:
        pockets = ", ".join(sig_gpt["group"].tolist())
        bottom_parts.append(f" GPT-4o's clearest statistically robust pockets are {pockets}.")
    if not sig_lang.empty:
        lang_qwen = sig_lang[sig_lang["leader"] == MODEL_B_LABEL]
        lang_gpt = sig_lang[sig_lang["leader"] == "gpt-4o"]
        if not lang_qwen.empty:
            bottom_parts.append(f" {MODEL_B_DISPLAY} is significantly ahead in {', '.join(lang_qwen['group'].tolist())}.")
        if not lang_gpt.empty:
            bottom_parts.append(f" GPT-4o is significantly ahead in {', '.join(lang_gpt['group'].tolist())}.")
    gap_pp = abs(overall['qwen_share'] - 0.5) * 100
    if gap_pp < 5:
        bottom_parts.append(" The overall gap is narrow.")
    elif gap_pp < 15:
        bottom_parts.append(" The overall gap should be viewed as moderate, not dominant.")
    else:
        bottom_parts.append(" The overall gap is substantial.")
    story.append(Paragraph("".join(bottom_parts), styles["body"]))

    def draw_header_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#6B7280"))
        canvas.drawString(doc.leftMargin, 0.35 * inch, REPORT_TITLE)
        canvas.drawRightString(letter[0] - doc.rightMargin, 0.35 * inch, f"Page {doc.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=draw_header_footer, onLaterPages=draw_header_footer)


def main() -> None:
    args = parse_args()
    args.asset_dir.mkdir(parents=True, exist_ok=True)
    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)

    df = load_data(args.judgments)
    overall_summary = exact_summary(
        int((df["winner"] == MODEL_B_LABEL).sum()),
        int((df["winner"] == "gpt-4o").sum()),
    )
    overall = {
        "total_n": int(len(df)),
        "qwen_wins": overall_summary.wins,
        "gpt_wins": overall_summary.losses,
        "ties": int((df["winner"] == "Tie").sum()),
        "decisive_n": overall_summary.n,
        "qwen_share": overall_summary.share,
        "qwen_ci_low": overall_summary.ci_low,
        "qwen_ci_high": overall_summary.ci_high,
        "qwen_p_value": overall_summary.p_value,
    }

    genre_table = subgroup_table(df, "genre")
    language_table = subgroup_table(df, "language", min_total_n=1)
    knowledge_table = subgroup_table(df, "knowledge")
    reasoning_table = subgroup_table(df, "reasoning_difficulty")
    prompt_length_table = subgroup_table(df, "prompt_length_bucket")
    position_stats = build_position_stats(df)
    length_stats = build_length_stats(df)
    cost_stats = build_cost_stats(df)
    reason_patterns = build_reason_pattern_table(df)

    if language_table.empty:
        major_languages = language_table.copy()
    else:
        major_languages = language_table[language_table["total_n"] >= 1].copy()
    difficulty_combo = pd.concat(
        [
            knowledge_table.assign(group=knowledge_table["group"].map(lambda value: f"Knowledge: {value}")),
            reasoning_table.assign(group=reasoning_table["group"].map(lambda value: f"Reasoning: {value}")),
            prompt_length_table.assign(group=prompt_length_table["group"].map(lambda value: f"Prompt length: {value}")),
        ],
        ignore_index=True,
    )

    asset_paths = {
        "overall": args.asset_dir / "overall_outcomes.png",
        "genre": args.asset_dir / "genre_forest.png",
        "language": args.asset_dir / "language_forest.png",
        "difficulty": args.asset_dir / "difficulty_forest.png",
        "length": args.asset_dir / "length_effect.png",
    }
    plot_overall_outcomes(df, asset_paths["overall"])
    plot_forest(genre_table.sort_values("qwen_share", ascending=False), "Genre pockets", asset_paths["genre"], max_rows=12)
    plot_forest(major_languages.sort_values("qwen_share", ascending=False), "Language pockets (10+ prompts)", asset_paths["language"], max_rows=12)
    plot_forest(difficulty_combo.sort_values("qwen_share", ascending=False), "Difficulty and prompt-length pockets", asset_paths["difficulty"], max_rows=12)
    plot_two_group_share(length_stats, asset_paths["length"])

    _, chi2_genre_p, _, _ = chi2_contingency(pd.crosstab(df[df["decisive"]]["genre"], df[df["decisive"]]["winner"]))
    _, chi2_language_p, _, _ = chi2_contingency(pd.crosstab(df[(df["decisive"]) & (df["language"].isin(major_languages["group"]))]["language"], df[(df["decisive"]) & (df["language"].isin(major_languages["group"]))]["winner"]))

    analysis = {
        "overall": overall,
        "position_stats": position_stats,
        "length_stats": length_stats,
        "cost_stats": cost_stats,
        "association_tests": {
            "genre_chi2_p": float(chi2_genre_p),
            "language_chi2_p": float(chi2_language_p),
        },
        "top_qwen_subgroups": genre_table[(genre_table["significant"]) & (genre_table["leader"] == MODEL_B_LABEL)].to_dict(orient="records"),
        "top_gpt_subgroups": genre_table[(genre_table["significant"]) & (genre_table["leader"] == "gpt-4o")].to_dict(orient="records"),
        "major_language_results": major_languages.to_dict(orient="records"),
        "knowledge_results": knowledge_table.to_dict(orient="records"),
        "reasoning_results": reasoning_table.to_dict(orient="records"),
        "prompt_length_results": prompt_length_table.to_dict(orient="records"),
        "reason_pattern_results": reason_patterns.to_dict(orient="records"),
    }
    args.output_json.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    build_pdf(
        pdf_path=args.output_pdf,
        asset_paths=asset_paths,
        analysis=analysis,
        genre_table=genre_table,
        language_table=major_languages,
        knowledge_table=knowledge_table,
        reasoning_table=reasoning_table,
        prompt_length_table=prompt_length_table,
        reason_patterns=reason_patterns,
    )
    print(args.output_pdf)
    print(args.output_json)


if __name__ == "__main__":
    main()
