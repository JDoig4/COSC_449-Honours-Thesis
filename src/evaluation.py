"""score the baselines with the metrics the deliverable asks for

run from the project root:  .venv/Scripts/python.exe src/evaluation.py
"""

# these are the scoring tools from scikit-learn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


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


if __name__ == "__main__":
    # load the prediction results
    from run_baselines import behaviour, results

    # score the overall majority baseline
    print(f"=== {behaviour}: overall majority ===")
    evaluate(results["label"], results["overall_prediction"])

    # score the personal majority baseline
    print(f"\n=== {behaviour}: personal majority ===")
    evaluate(results["label"], results["personal_prediction"])
