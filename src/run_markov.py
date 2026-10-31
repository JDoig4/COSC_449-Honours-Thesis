"""score the Markov models on the held-out test set and update the results graphs

uses the same train/test split as Deliverable 1, so the D1 baselines and the Markov models are
scored on exactly the same test utterances.

run from the project root:  .venv/Scripts/python.exe src/run_markov.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

import markov_models as mm
from evaluation import plot_confusion
from markov_plots import GRID, INK, MODEL_STYLE, SECONDARY, SEQUENTIAL, plot_matrix, save
from run_baselines import load_data, run_behaviour

MODELS = ["overall_frequency", "personal_frequency", "overall_markov", "personal_markov"]

# the two smoothing options (see markov_models.shrink). normalized is the report's main one.
SMOOTHING = ["normalized", "none"]
SMOOTHING_LABEL = {"normalized": "normalized smoothing", "none": "no smoothing"}


def choose_strengths(train, codes, smoothing):
    """How strongly each Markov model is pulled toward its prior.

    "normalized": picked by cross-validation on train only. also returns the score of every
    strength tried, for the strength-choice figure.
    "none": 0 for both, so every row is its raw proportions.
    the two frequency models have no strength: neither is ever smoothed.
    """
    if smoothing == "none":
        return {"overall_markov": 0, "personal_markov": 0}, None

    population_strength, personal_strength, strengths = mm.choose_markov_strengths(train, codes)
    chosen = {"overall_markov": population_strength, "personal_markov": personal_strength}
    return chosen, strengths


def test_probabilities(train, test, codes, chosen):
    """Give every test utterance a chance for each code, from each of the four models."""
    return {
        "overall_frequency": mm.predict_overall_frequency(train, test, codes),
        "personal_frequency": mm.predict_personal_frequency(train, test, codes),
        "overall_markov": mm.predict_overall_markov(train, test, codes, chosen["overall_markov"]),
        "personal_markov": mm.predict_personal_markov(
            train, test, codes, chosen["personal_markov"], population_strength=chosen["overall_markov"]),
    }


def next_code_scores(probabilities, actual, codes, order):
    """Score the 14-way task: predict which code an utterance has."""
    rows = []
    for name in MODELS:
        probs = probabilities[name]

        # the guess is the code with the highest chance (a tie goes to the more common code)
        guess = mm.top_code(probs, order)
        bits = mm.log_loss_bits(probs, actual)
        rows.append({
            "model": name,
            "cross_entropy_bits": bits,
            "perplexity": 2 ** bits,
            # share of utterances whose real code had a chance of 0, standing in as EPSILON. each of
            # these adds log2(1 / EPSILON), about 33 bits, to the cross-entropy sum
            "epsilon_share": float((mm.chance_of_actual(probs, actual) == mm.EPSILON).mean()),
            "accuracy": accuracy_score(actual, guess),
            "macro_f1": f1_score(actual, guess, labels=codes, average="macro", zero_division=0),
        })
    return pd.DataFrame(rows)


def per_code_scores(probabilities, actual, codes, order):
    """Precision, recall and F1 for every code on the 14-way task, for every model.

    macro-F1 is the plain average of these 14 F1 scores, so this table shows where it comes from.
    """
    tables = []
    for name in MODELS:
        guess = mm.top_code(probabilities[name], order)
        precision, recall, f1, support = precision_recall_fscore_support(
            actual, guess, labels=codes, zero_division=0)
        tables.append(pd.DataFrame({
            "model": name,
            "code": codes,
            "test_utterances": support,
            "guessed": guess.value_counts().reindex(codes, fill_value=0).values,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }))
    return pd.concat(tables, ignore_index=True)


def behaviour_scores(df, probabilities, codes, smoothing):
    """Score the 14 yes/no tasks from D1, with the D1 baselines and the two Markov models.

    a Markov model says yes (1) when it gives the behaviour more than a 50% chance. that is the
    same rule the D1 majority baselines use, but now the chance depends on the previous code.
    the D1 baselines use raw proportions, so they are the same under both smoothing options.
    """
    # the confusion pictures for the no-smoothing option get their own file names
    suffix = "" if smoothing == "normalized" else f"_{smoothing}"
    rows = []
    for behaviour in codes:
        _, _, _, _, results = run_behaviour(df, behaviour)
        predictions = {
            "overall_majority": results["overall_prediction"],
            "personal_majority": results["personal_prediction"],
            "overall_markov": (probabilities["overall_markov"][behaviour] > 0.5).astype(int),
            "personal_markov": (probabilities["personal_markov"][behaviour] > 0.5).astype(int),
        }
        for name, predicted in predictions.items():
            predicted = predicted.loc[results.index]

            # the whole confusion matrix (real no/yes against predicted no/yes) and the
            # per-class scores. class 0 = "no, not this behaviour", class 1 = "yes, this behaviour"
            tn, fp, fn, tp = confusion_matrix(results["label"], predicted, labels=[0, 1]).ravel()
            precision, recall, f1, _ = precision_recall_fscore_support(
                results["label"], predicted, labels=[0, 1], zero_division=0)
            rows.append({
                "behaviour": behaviour,
                "model": name,
                "accuracy": accuracy_score(results["label"], predicted),
                "macro_f1": f1_score(results["label"], predicted, labels=[0, 1],
                                     average="macro", zero_division=0),
                "predicted_yes": int(predicted.sum()),
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "precision_no": precision[0], "recall_no": recall[0], "f1_no": f1[0],
                "precision_yes": precision[1], "recall_yes": recall[1], "f1_yes": f1[1],
            })

            # same three-panel confusion pictures as D1, for the new models
            if name.endswith("markov"):
                plot_confusion(results["label"], predicted,
                               f"{behaviour}: {MODEL_STYLE[name]['label']} ({SMOOTHING_LABEL[smoothing]})",
                               f"outputs/{behaviour}_{name}{suffix}.png", behaviour)
    return pd.DataFrame(rows)


def style_axis(ax):
    """Light, recessive axes: no box, hairline grid behind the marks."""
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)


def plot_next_code_results(scores):
    """Three small bar charts side by side: cross-entropy, accuracy, macro-F1 for the 14-way task."""
    measures = [("cross_entropy_bits", "Cross-entropy (bits, lower is better)"),
                ("accuracy", "Accuracy"),
                ("macro_f1", "Macro-F1 (14 codes)")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.6), sharey=True)
    y = np.arange(len(MODELS))[::-1]

    for ax, (column, title) in zip(axes, measures):
        values = scores.set_index("model").loc[MODELS, column]
        colors = [MODEL_STYLE[name]["color"] for name in MODELS]
        ax.barh(y, values, height=0.6, color=colors)
        for yi, value in zip(y, values):
            ax.text(value, yi, f" {value:.2f}", va="center", fontsize=8, color=SECONDARY)
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_xlim(0, values.max() * 1.25)
        style_axis(ax)

    axes[0].set_yticks(y, [MODEL_STYLE[name]["label"] for name in MODELS])
    fig.suptitle("Predicting each held-out utterance's code", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "results_next_code.png")


def plot_behaviour_results(scores, counts):
    """Dot plot: macro-F1 on each D1 yes/no task, one marker per model."""
    order = counts.sort_values().index  # most common behaviour at the top
    names = ["overall_majority", "personal_majority", "overall_markov", "personal_markov"]
    table = scores.pivot(index="behaviour", columns="model", values="macro_f1").loc[order]

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    y = np.arange(len(order))
    offsets = np.linspace(-0.24, 0.24, len(names))
    for offset, name in zip(offsets, names):
        style = MODEL_STYLE[name]
        ax.scatter(table[name], y + offset, s=34, color=style["color"], marker=style["marker"],
                   label=style["label"], edgecolors="white", linewidths=0.6, zorder=3)

    # the score of a model that always says "no": half of a perfect macro-F1 on the "no" class
    ax.axvline(0.5, color=SECONDARY, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
    ax.text(0.5, len(order) - 0.35, " always 'no' ≤ 0.5", fontsize=7.5, color=SECONDARY, va="bottom")

    ax.set_yticks(y, [f"{b} ({counts[b]:,})" for b in order])
    ax.set_xlabel("macro-F1 on the test set")
    ax.set_title("D1 yes/no tasks, updated with the Markov models", loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    style_axis(ax)
    fig.tight_layout()
    save(fig, "results_behaviours.png")


def plot_behaviour_confusions(scores, name, order, smoothing="normalized"):
    """All 14 yes/no confusion matrices for one model, one small panel per behaviour.

    rows are the real answer, columns the model's answer. the number is a count of test
    utterances. the shading is that count's share of its row, so the shade of the bottom-right
    cell is the recall for "yes".
    """
    scores = scores[scores["smoothing"] == smoothing]
    label = MODEL_STYLE[name]["label"]
    if smoothing != "normalized" and name.endswith("markov"):
        label += f" ({SMOOTHING_LABEL[smoothing]})"
    table = scores[scores["model"] == name].set_index("behaviour")
    fig, axes = plt.subplots(3, 5, figsize=(10, 6.6))

    for ax, behaviour in zip(axes.flat, order):
        row = table.loc[behaviour]
        counts = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
        shares = counts / counts.sum(axis=1, keepdims=True)
        ax.imshow(shares, cmap=SEQUENTIAL, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{counts[i, j]:,}", ha="center", va="center", fontsize=8,
                        color="white" if shares[i, j] > 0.6 else INK)
        ax.set_xticks([0, 1], ["no", "yes"])
        ax.set_yticks([0, 1], ["no", "yes"])
        ax.set_title(f"{behaviour} ({row['fn'] + row['tp']:,} real)", loc="left", fontsize=8.5)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # 15 panels, 14 behaviours: the last one is left empty
    for ax in axes.flat[len(order):]:
        ax.axis("off")

    fig.supylabel("real: does the utterance have this code?", color=SECONDARY, fontsize=9)
    fig.supxlabel("predicted: does the model say it has this code?", color=SECONDARY, fontsize=9)
    fig.suptitle(f"{label}: yes/no confusion matrices for all 14 codes",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    suffix = "" if smoothing == "normalized" else f"_{smoothing}"
    save(fig, f"confusion_yes_no_{name}{suffix}.png")


def plot_next_code_confusion(probabilities, actual, codes, order):
    """Which codes the personal Markov model confuses: rows = real code, each row adds to 1."""
    guess = mm.top_code(probabilities["personal_markov"], order)
    matrix = pd.DataFrame(confusion_matrix(actual, guess, labels=codes, normalize="true"),
                          index=codes, columns=codes)

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    plot_matrix(ax, matrix, "Personal Markov: real code (rows) vs guessed code (columns)", vmax=1)
    ax.set_xlabel("guessed code")
    ax.set_ylabel("real code")
    fig.tight_layout()
    save(fig, "results_confusion_personal_markov.png")


def plot_strength_choice(strengths, chosen):
    """Cross-entropy for every shrinkage strength tried on the training folds."""
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for name in ["overall_markov", "personal_markov"]:
        style = MODEL_STYLE[name]
        ax.plot(strengths.index, strengths[name], color=style["color"], marker=style["marker"],
                markersize=5, linewidth=2, label=f"{style['label']} (chosen {chosen[name]:g})")
    ax.set_xscale("log")
    ax.set_xlabel("strength (pretend observations of the prior)")
    ax.set_ylabel("cross-entropy (bits)")
    ax.set_title("Choosing how much to pull toward the prior (5-fold CV on train)", loc="left")
    ax.legend(frameon=False, fontsize=8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.grid(color=GRID, linewidth=0.8)
    fig.tight_layout()
    save(fig, "strength_choice.png")


if __name__ == "__main__":
    # same split as D1, then give every utterance its previous code inside its PR
    df = mm.add_previous_code(load_data())
    codes = sorted(df["Code"].unique())
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]

    starts = (test["previous"] == mm.START).mean()
    print(f"train {len(train)} / test {len(test)} utterances; {starts:.0%} of test utterances open a PR sequence")

    # when two codes tie for the highest chance, the guess is the one more common in training
    order = train["Code"].value_counts().index

    # score every model under both smoothing options, on the same test utterances
    next_code, per_code, behaviours = [], [], []
    for smoothing in SMOOTHING:
        chosen, strengths = choose_strengths(train, codes, smoothing)
        probabilities = test_probabilities(train, test, codes, chosen)
        print(f"\n##### {smoothing} smoothing: strengths {chosen}")
        if strengths is not None:
            print(strengths.round(3).to_string())

        # every model's rows should add up to 1
        for name, probs in probabilities.items():
            assert np.allclose(probs.sum(axis=1), 1), name

        next_code.append(next_code_scores(probabilities, test["Code"], codes, order).assign(smoothing=smoothing))
        per_code.append(per_code_scores(probabilities, test["Code"], codes, order).assign(smoothing=smoothing))
        behaviours.append(behaviour_scores(df, probabilities, codes, smoothing).assign(smoothing=smoothing))

        # these two figures are for the main (normalized) option only
        if smoothing == "normalized":
            plot_next_code_confusion(probabilities, test["Code"], codes, order)
            plot_strength_choice(strengths, chosen)

    next_code = pd.concat(next_code, ignore_index=True)
    print("\n=== 14-way next-code prediction (test set) ===")
    print(next_code.round(3).to_string(index=False))
    next_code.round(4).to_csv("outputs/markov_next_code.csv", index=False)

    per_code = pd.concat(per_code, ignore_index=True)
    print("\n=== 14-way task, F1 for every code (test set) ===")
    print(per_code.pivot(index="code", columns=["smoothing", "model"], values="f1").round(2).to_string())
    per_code.round(4).to_csv("outputs/markov_per_code.csv", index=False)

    behaviours = pd.concat(behaviours, ignore_index=True)
    print("\n=== D1 yes/no tasks, macro-F1 (test set) ===")
    print(behaviours.pivot(index="behaviour", columns=["smoothing", "model"], values="macro_f1").round(3).to_string())
    print("\nhow often each Markov model says yes:")
    print(behaviours[behaviours["model"].str.endswith("markov")]
          .pivot(index="behaviour", columns=["smoothing", "model"], values="predicted_yes").to_string())
    behaviours.round(4).to_csv("outputs/markov_behaviours.csv", index=False)

    plot_next_code_results(next_code[next_code["smoothing"] == "normalized"])
    plot_behaviour_results(behaviours[behaviours["smoothing"] == "normalized"], df["Code"].value_counts())
    by_frequency = df["Code"].value_counts().index
    for name in ["personal_majority", "personal_markov"]:
        plot_behaviour_confusions(behaviours, name, by_frequency)
    plot_behaviour_confusions(behaviours, "personal_markov", by_frequency, smoothing="none")
    print("\nfigures saved to report/figures/, confusion matrices to outputs/")
