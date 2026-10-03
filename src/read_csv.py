import glob
import os
import re

import pandas as pd


def load_reviews():
    """read all CSVs and create data frame for team name"""
    dfs = []
    for path in sorted(glob.glob("data/team-*-review-comments.csv")):
        team = re.search(r"team-(\d+)-", os.path.basename(path)).group(1)
        df = pd.read_csv(path)
        df["team"] = team
        dfs.append(df)


    # concatenate all reviews by team
    all_reviews = pd.concat(dfs, ignore_index=True)


    #fix continuity error for label
    all_reviews["Code"] = all_reviews["Code"].str.strip().replace({"updating": "Updating"})


    #normalize timestamps to UTC (source offsets mix -07:00 / -08:00 across PST/PDT)
    all_reviews["created_at"] = pd.to_datetime(all_reviews["created_at"], utc=True)
    all_reviews["updated_at"] = pd.to_datetime(all_reviews["updated_at"], utc=True)


    return all_reviews.sort_values("created_at", ignore_index=True, kind="mergesort")

def dedupe_frame(df):
    """Keep one row for each unique comment ID and code combination.

    The data does not include the text or location of individual utterances,
    so we cant tell why the same code appears more than once on a single comment.
    These rows might represent separate utterances or could be duplicate records.
    """
    return df.drop_duplicates(subset=["comment_id", "Code"], ignore_index=True, keep="first")


def binary_labels(df, behaviour):
    """Return a 0/1 label for one behaviour on every row (utterance) of df.

    Each row is one utterance with exactly one Code, and repeated rows are separate
    instances: two Directing statements in one comment are two rows, and both get a 1.
    Nothing is merged or collapsed, so the result lines up one-to-one with df's rows and
    keeps df's index.

    A row with no Code at all is an unannotated row, not a behaviour that is absent, so it
    is refused instead of being quietly labelled 0. A behaviour name that never appears in
    the data (most likely a typo) is refused too, since it would label every row 0.
    """
    if df["Code"].isna().any():
        raise ValueError(f"{df['Code'].isna().sum()} rows have no Code")
    if behaviour not in set(df["Code"]):
        raise ValueError(f"behaviour {behaviour!r} does not appear in the Code column")

    return (df["Code"] == behaviour).astype(int).rename(behaviour)

if __name__ == "__main__":
    reviews = load_reviews()
    reviews.info()