import pandas as pd

import baselines, read_csv, split_data


def load_data():
    """Load the reviews and split them into train and test."""

    # read the csvs and give every row a train or test label
    reviews = read_csv.load_reviews()
    return split_data.make_split(reviews)


def run_behaviour(df, behaviour):
    """Run both baselines for one behaviour and return the pieces."""

    # work on a copy so the label column doesn't carry over between behaviours
    df = df.copy()

    # 1 if the row is this behaviour, 0 if not
    df["label"] = read_csv.binary_labels(df, behaviour)

    # separate train and test rows
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]

    # learn both baselines from train only
    overall = baselines.overall_majority(train, "label")
    personal = baselines.personal_majorities(train, "label", overall)

    # make predictions for the test set
    overall_prediction = baselines.predict_overall(test, overall)
    personal_prediction = baselines.predict_personal(test, personal, overall)

    # put the real label and both guesses side by side
    results = pd.concat(
        [test[["comment_id", "commenter", "label"]], overall_prediction, personal_prediction],
        axis=1,
    )

    return train, test, overall, personal, results


if __name__ == "__main__":
    # quick look at one behaviour
    train, test, overall, personal, results = run_behaviour(load_data(), "Approving")
    print(f"Overall majority for Approving: {overall}")
    print(results.head(10).to_string())
