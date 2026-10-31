"""an overview of who wrote the utterances: how much data each commenter and each team has

prints the numbers for the report's data section and draws one figure.

run from the project root:  .venv/Scripts/python.exe src/data_overview.py
"""

import matplotlib.pyplot as plt
import pandas as pd

import markov_models as mm
import read_csv
from markov_plots import GRID, SECONDARY, save

BAR = "#2a78d6"


def per_commenter_table(df):
    """How much data each commenter has, summarised as min / quartiles / max."""
    sequences = df.groupby(["team", "pr_id", "commenter"]).size().groupby("commenter").size()
    table = pd.DataFrame({
        "utterances": df.groupby("commenter").size(),
        "comments": df.groupby("commenter")["comment_id"].nunique(),
        "pull requests": sequences,  # one sequence per PR a commenter wrote in
        "codes used (of 14)": df.groupby("commenter")["Code"].nunique(),
        "within-PR transitions": df[df["previous"] != mm.START].groupby("commenter").size(),
    }).fillna(0)
    return table.describe(percentiles=[0.25, 0.5, 0.75]).loc[["min", "25%", "50%", "75%", "max"]]


def plot_overview(df):
    """Left: utterances per commenter, largest first. Right: utterances per team, largest first."""
    per_commenter = df.groupby("commenter").size().sort_values(ascending=False)
    per_team = df.groupby("team").size().sort_values(ascending=False)

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1.7, 1]})

    # a log scale, because the busiest commenter has 700 times as many utterances as the quietest
    left.bar(range(len(per_commenter)), per_commenter.values, width=0.8, color=BAR)
    left.set_yscale("log")
    median = per_commenter.median()
    left.axhline(median, color=SECONDARY, linewidth=0.8, linestyle=(0, (3, 3)))
    left.text(len(per_commenter) - 1, median, f"median {median:.0f}", ha="right", va="bottom",
              fontsize=8, color=SECONDARY)
    left.set_xticks([])
    left.set_xlabel(f"the {len(per_commenter)} commenters, most utterances first")
    left.set_ylabel("utterances (log scale)")
    left.set_title("Utterances per commenter", loc="left")

    right.bar(range(len(per_team)), per_team.values, width=0.8, color=BAR)
    right.set_xticks(range(len(per_team)), per_team.index, fontsize=7)
    right.set_xlabel("team number")
    right.set_ylabel("utterances")
    right.set_title("Utterances per team", loc="left")

    for ax in (left, right):
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)

    fig.tight_layout()
    save(fig, "data_overview.png")


if __name__ == "__main__":
    df = mm.add_previous_code(read_csv.load_reviews())

    print("=== per commenter ===")
    print(per_commenter_table(df).round(0).astype(int).to_string())

    per_commenter = df.groupby("commenter").size()
    print(f"\ncommenters with fewer than 30 utterances: {(per_commenter < 30).sum()}")
    print(f"share of utterances from the 10 busiest commenters: {per_commenter.nlargest(10).sum() / len(df):.0%}")
    print(f"commenters in more than one team: {(df.groupby('commenter')['team'].nunique() > 1).sum()}")
    print(f"utterances written on the commenter's own PR: {(df['commenter'] == df['pr_author']).mean():.0%}")

    per_team = df.groupby("team")["commenter"].nunique()
    print(f"commenters per team: min {per_team.min()}, median {per_team.median():.0f}, max {per_team.max()}")

    per_comment = df.groupby("comment_id").size()
    print(f"comments with one utterance: {(per_comment == 1).mean():.0%}, mean utterances per comment: "
          f"{per_comment.mean():.2f}, most: {per_comment.max()}")

    plot_overview(df)
    print("\nfigure saved to report/figures/")
