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


if __name__ == "__main__":
    reviews = load_reviews()
    reviews.info()