"""tests for binary_labels, plus checks that the utterance-level data and split are sound

run from the project root:  .venv/Scripts/python.exe -m pytest -v
"""

import pandas as pd
import pytest

from read_csv import binary_labels, load_reviews
from split_data import make_split


# --- layer 1: a tiny hand-built frame where the right answer is known ------

def toy_long():
    # comment 101 has two Directing utterances: both must stay, and both must be 1.
    # the index is deliberately not 0..n so the test catches any accidental reset
    return pd.DataFrame(
        {
            "comment_id": [101, 101, 101, 102],
            "commenter":  ["alice", "alice", "alice", "bob"],
            "Code":       ["Directing", "Directing", "Approving", "Empty"],
        },
        index=[7, 3, 9, 5],
    )


def test_repeated_utterances_are_each_labelled():
    expected = pd.Series([1, 1, 0, 0], index=[7, 3, 9, 5], name="Directing")
    pd.testing.assert_series_equal(binary_labels(toy_long(), "Directing"), expected)


def test_other_behaviour_on_the_same_rows():
    expected = pd.Series([0, 0, 1, 0], index=[7, 3, 9, 5], name="Approving")
    pd.testing.assert_series_equal(binary_labels(toy_long(), "Approving"), expected)


def test_input_frame_is_not_changed():
    long = toy_long()
    before = long.copy()
    binary_labels(long, "Directing")
    pd.testing.assert_frame_equal(long, before)


def test_unknown_behaviour_is_rejected():
    # a typo like "Directng" would otherwise label every row 0 without complaint
    with pytest.raises(ValueError, match="Directng"):
        binary_labels(toy_long(), "Directng")


def test_missing_code_is_rejected_not_treated_as_zero():
    long = toy_long()
    long.loc[9, "Code"] = None
    with pytest.raises(ValueError, match="no Code"):
        binary_labels(long, "Directing")


# --- layer 2: checks on the real data --------------------------------------

@pytest.fixture(scope="module")
def reviews():
    return load_reviews()


@pytest.fixture(scope="module")
def split(reviews):
    return make_split(reviews)


def test_every_row_has_exactly_one_code(reviews):
    assert reviews["Code"].notna().all()


def test_label_totals_match_raw_row_counts(reviews):
    # repeats are counted, so each behaviour's total is its raw number of rows,
    # e.g. 2075 for Directing, not the number of distinct comments
    for behaviour, count in reviews["Code"].value_counts().items():
        labels = binary_labels(reviews, behaviour)
        assert len(labels) == len(reviews)
        assert labels.sum() == count, behaviour


def test_each_row_is_positive_for_exactly_one_behaviour(reviews):
    # since every row has one Code, summing the labels across all behaviours gives 1 per row
    all_labels = pd.concat(
        [binary_labels(reviews, b) for b in reviews["Code"].unique()], axis=1
    )
    assert (all_labels.sum(axis=1) == 1).all()


def test_each_comment_has_one_commenter(reviews):
    assert reviews.groupby("comment_id")["commenter"].nunique().max() == 1


def test_split_keeps_every_row(reviews, split):
    assert len(split) == len(reviews)
    assert split["split"].isin(["train", "test"]).all()


def test_no_comment_on_both_sides_of_the_split(split):
    train_ids = set(split.loc[split["split"] == "train", "comment_id"])
    test_ids = set(split.loc[split["split"] == "test", "comment_id"])
    assert train_ids.isdisjoint(test_ids)
