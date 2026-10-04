"""sanity checks on the baselines built in run_baselines.py

run from the project root:  .venv/Scripts/python.exe src/sanity_checks.py
"""

import pandas as pd

from run_baselines import load_data, run_behaviour

# load and split the data once
df = load_data()

# run the checks on approving
behaviour = "Approving"
train, test, overall, personal, results = run_behaviour(df, behaviour)


print(f"{behaviour} share of training rows: {train['label'].mean()}")
print(f"Overall majority: {overall}")

#ties: commenters with a training mean of exactly 0.5, who default to the overall majority
means = train.groupby("commenter")["label"].mean()
ties = means[means == 0.5]
print(f"Tied commenters: {len(ties)}, their test rows: {test['commenter'].isin(ties.index).sum()}")

#unseen: test commenters with no training rows, who also default to the overall majority
unseen = ~test["commenter"].isin(personal)
print(f"Unseen commenters: {test.loc[unseen, 'commenter'].nunique()}, their test rows: {unseen.sum()}")

#hand check: pick three real commenters and make sure the code followed our rule for each.
#we use one person the code predicted 1 for, one it predicted 0 for, and one of the ties.
#for the tie, we take the first tied person who actually has rows in the test set, since
#a tied person with no test rows would have no predictions for us to look at.
tied_in_test = [name for name in ties.index if name in set(test["commenter"])]
people_to_check = ["bcalderon", "ekaiser", tied_in_test[0]]

for name in people_to_check:
    #this is the person's share of approving rows in the training data
    mean = means[name]

    #this is what our rule says they should get: above half means 1, below half means 0,
    #and exactly half is a tie, so they get the overall majority instead
    if mean > 0.5:
        expected = 1
    elif mean < 0.5:
        expected = 0
    else:
        expected = overall

    #these are the predictions the code actually made for this person's test rows.
    #every row for one person should have the same prediction, so we expect one value here
    predicted = results.loc[results["commenter"] == name, "personal_prediction"].unique()

    print(f"{name}: training mean {mean:.3f}, rule says {expected}, code predicted {predicted}")

#rough accuracy: the share of test rows where the guess matches the real label.
#comparing the two columns gives True for a right guess and False for a wrong one, and since
#True counts as 1, taking the mean gives the share of right guesses.
#the overall baseline always guesses 0, so it should be right on every non-approving row (about 77%)
print(f"Overall accuracy: {(results['label'] == results['overall_prediction']).mean():.3f}")
print(f"Personal accuracy: {(results['label'] == results['personal_prediction']).mean():.3f}")

#class sizes: how many rows of each behaviour ended up in train and in test.
#crosstab counts the rows for every behaviour and split combination
sizes = pd.crosstab(df["Code"], df["split"])

#share of each behaviour's rows that landed in test (should be close to 0.5 for all of them)
sizes["test_share"] = (sizes["test"] / (sizes["train"] + sizes["test"])).round(3)

print("\nClass sizes per split:")
print(sizes.to_string())
