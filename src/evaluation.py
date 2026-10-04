"""score the baselines with the metrics the deliverable asks for

run from the project root:  .venv/Scripts/python.exe src/evaluation.py
"""

import os

import matplotlib.pyplot as plt

# these are the scoring tools from scikit-learn
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix, f1_score
)


def evaluate(actual, predicted):
    """Print every metric for one set of predictions."""

    # show how often the predictions were correct
    print(f"Accuracy: {accuracy_score(actual, predicted):.3f}")

    # show the average F1 score for both classes
    # include both classes and use 0 when a score cannot be calculated
    print(f"Macro-F1: {f1_score(actual, predicted, labels=[0, 1], average='macro', zero_division=0):.3f}")

    # show precision, recall, and F1 for each class
    print(classification_report(actual, predicted, labels=[0, 1], zero_division=0))

    # show the number of correct and incorrect predictions for each class
    print("Confusion matrix:")
    print(confusion_matrix(actual, predicted, labels=[0, 1]))



def plot_confusion(actual, predicted, title, path):
    """Save one picture with three confusion matrices side by side: counts, recall, precision."""

    # 3 panel plot
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Counts: shows the plain numbers
    # Recall: shows what share of the real 0s and real 1s were guessed right
    # Precision: shows what share of the 0 guesses and 1 guesses were right
    panels = [("Counts", None), ("Recall (rows add to 1)", "true"), ("Precision (columns add to 1)", "pred")]

    # draw each panel
    for ax, (name, normalize) in zip(axes, panels):
        # whole nums for counts, decimal for recall and precision
        number_format = "d" if normalize is None else ".2f"
        # draw the grid of real label against guess
        ConfusionMatrixDisplay.from_predictions(
            actual, predicted, labels=[0, 1], normalize=normalize,
            values_format=number_format, cmap="Blues", colorbar=False, ax=ax,
        )
        # name the panel
        ax.set_title(name)

    # put title on top and set layout to tight
    fig.suptitle(title)
    fig.tight_layout()

    # make output folder if it doesnt exist and save
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)

    # close fig
    plt.close(fig)


if __name__ == "__main__":
    # load the prediction results
    from run_baselines import behaviour, results

    # score the overall majority baseline
    print(f"=== {behaviour}: overall majority ===")
    evaluate(results["label"], results["overall_prediction"])
    plot_confusion(results["label"], results["overall_prediction"],
                   f"{behaviour}: overall majority", f"outputs/{behaviour}_overall.png")

    # score the personal majority baseline
    print(f"\n=== {behaviour}: personal majority ===")
    evaluate(results["label"], results["personal_prediction"])
    plot_confusion(results["label"], results["personal_prediction"],
                   f"{behaviour}: personal majority", f"outputs/{behaviour}_personal.png")
