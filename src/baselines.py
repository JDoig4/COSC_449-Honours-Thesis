import read_csv, split_data
import pandas as pd

def overall_majority(train, label_col):
    """Return the majority label for the training data.

    This model predicts 1 when more than half of the training labels are 1.
    Otherwise, it predicts 0. If the labels are exactly tied, it predicts 0.
    """

    if train[label_col].mean() > 0.5:
        return 1
    else:
        return 0


def personal_majorities(train, label_col, overall):
    """Return each commenter's most common label for the given behaviour.

    For each commenter, predict 1 when more than half of their training labels are 1,
    and predict 0 when fewer than half are 1. If the labels are tied, use the overall
    majority label.
    """

    def decide(mean):
        if mean > 0.5:
            return 1
        if mean < 0.5:
            return 0
        return overall

    # compute the mean label for each commenter and apply the decision function
    personal = train.groupby("commenter")[label_col].mean()
    # apply the decision function to each commenter's mean label to determine their personal majority
    personal = personal.apply(decide)

    return personal.to_dict()

def predict_personal(test, personal, overall):
    """Return a Series of predictions for the test set based on personal majorities.

    if a commenter has no training data (would show as NaN), the overall majority is used instead.
    """
    return test["commenter"].map(personal).fillna(overall).astype(int).rename("personal_prediction")


def predict_overall(test, overall):
    """Return a Series of predictions for the test set based on the overall majority."""
    return pd.Series(overall, index=test.index, name="overall_prediction")


