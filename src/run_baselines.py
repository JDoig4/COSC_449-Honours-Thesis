import pandas as pd

import baselines, read_csv, split_data


behaviour = "Approving"

#establish the train/test split and the label column for this behaviour
reviews = read_csv.load_reviews()
df = split_data.make_split(reviews)

df["label"] = read_csv.binary_labels(df, behaviour)

train = df[df["split"] == "train"]
test = df[df["split"] == "test"]

overall = baselines.overall_majority(train, "label")
personal = baselines.personal_majorities(train, "label", overall)

# Make predictions for the test set
overall_prediction = baselines.predict_overall(test, overall)
personal_prediction = baselines.predict_personal(test, personal, overall)

results = pd.concat(
    [test[["comment_id", "commenter", "label"]], overall_prediction, personal_prediction],
    axis=1,
)


if __name__ == "__main__":
    print(f"Overall majority for {behaviour}: {overall}")
    print(results.head(10).to_string())
