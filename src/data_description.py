"""print the figures for the deliverable 1 data description

run from the project root:  .venv/Scripts/python.exe src/data_description.py > data_description.txt
"""

import pandas as pd

from read_csv import load_reviews


# load every row as it comes out of the csvs. each row is one utterance,
# and repeated rows are separate utterances, so nothing is removed
df = load_reviews()


# --- 1. size -----------------------------------------------------------------
# how many utterances (rows) per comment
per_comment = df.groupby("comment_id").size()

# how many rows repeat a code already used earlier in the same comment
repeats = df.duplicated(subset=["comment_id", "Code"]).sum()

print("=== size ===")
print("utterances (rows):             ", len(df))
print("comments:                      ", df["comment_id"].nunique())
print("comments with >1 utterance:    ", (per_comment > 1).sum())
print("most utterances in one comment:", per_comment.max())
print("repeated code in same comment: ", repeats)
print()


# --- 2. what the data covers -------------------------------------------------
# pr numbers restart in every team, so a pull request is a (team, pr_id) pair
print("=== corpus ===")
print("teams:         ", df["team"].nunique())
print("commenters:    ", df["commenter"].nunique())
print("pull requests: ", len(df.groupby(["team", "pr_id"])))
print("labels:        ", df["Code"].nunique())
print("first comment: ", df["created_at"].min())
print("last comment:  ", df["created_at"].max())
print()


# --- 3. label distribution ---------------------------------------------------
# how many utterances have each code, and what percent of all utterances that is
labels = df["Code"].value_counts().to_frame("utterances")
labels["percent"] = (100 * labels["utterances"] / len(df)).round(1)

print("=== label distribution ===")
print(labels.to_string())
print()


# --- 4. utterances per commenter ---------------------------------------------
# the personal baseline learns from each person's rows, so show how many each person has
per_person = df.groupby("commenter").size()

print("=== utterances per commenter ===")
print("fewest: ", per_person.min())
print("median: ", per_person.median())
print("most:   ", per_person.max())
