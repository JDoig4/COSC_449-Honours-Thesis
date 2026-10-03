from read_csv import load_reviews, dedupe_frame
from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd

def make_split(df, random_state = 41):
    """Split the data into train and test sets using stratified group k-fold.

    The split is stratified by the 'Code' column and grouped by 'comment_id' to ensure that
    all rows with the same comment_id are in the same split. The function returns a DataFrame
    with an additional 'split' column indicating the split assignment.
    """
    # work on a copy so the caller's frame is never given a 'split' column behind its back
    df = df.copy()

    sgkf = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=random_state)
    folds = sgkf.split(df, y=df["Code"], groups = df["comment_id"])

    # .split() yields one (train, test) pair per fold; take only the first, since
    # this is a single train/test split rather than a cross-validation run
    train_idx, test_idx = next(folds)

    # label each row by the split it landed in. the splitter returns positional
    # indices, so assignment goes through .iloc rather than .loc. starting from an
    # empty column means any row the splitter failed to place stays NaN and shows
    # up in the checks, instead of being silently counted as train
    split = pd.Series(index=df.index, dtype="object")
    split.iloc[train_idx] = "train"
    split.iloc[test_idx] = "test"
    df["split"] = split

    return df
