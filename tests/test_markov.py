"""tests for the Markov model code, on tiny hand-built frames where the right answer is known

run from the project root:  .venv/Scripts/python.exe -m pytest -v
"""

import numpy as np
import pandas as pd
import pytest

import markov_models as mm

CODES = ["Approving", "Directing", "Empty"]


def toy_reviews():
    # alice and bob both comment on PR 1 of team 1, taking turns.
    # alice also has PR 1 in team 2: same pr_id, but a different PR, so it must not link up.
    # alice's 2nd and 3rd rows are one comment (same timestamp), kept in file order.
    return pd.DataFrame({
        "created_at": pd.to_datetime(["2023-10-01 10:00", "2023-10-01 10:05", "2023-10-01 10:10",
                                      "2023-10-01 10:10", "2023-10-01 11:00", "2023-10-02 09:00"]),
        "team":       ["1", "1", "1", "1", "1", "2"],
        "pr_id":      [1, 1, 1, 1, 1, 1],
        "commenter":  ["alice", "bob", "alice", "alice", "bob", "alice"],
        "comment_id": [10, 11, 12, 12, 13, 14],
        "Code":       ["Directing", "Approving", "Directing", "Empty", "Approving", "Approving"],
    })


def test_previous_code_stays_inside_one_commenter_and_pr():
    previous = mm.add_previous_code(toy_reviews())["previous"].tolist()
    # bob's turns in between don't break alice's sequence; team 2's PR 1 starts fresh
    assert previous == [mm.START, mm.START, "Directing", "Directing", "Approving", mm.START]


def test_rows_out_of_time_order_are_refused():
    with pytest.raises(ValueError):
        mm.add_previous_code(toy_reviews().iloc[::-1])


def test_shrink_with_no_data_gives_the_prior():
    prior = np.array([0.2, 0.3, 0.5])
    result = mm.shrink([[0, 0, 0]], prior, strength=10)
    np.testing.assert_allclose(result[0], prior)


def test_shrink_with_lots_of_data_follows_the_data():
    result = mm.shrink([[1000, 0, 0]], np.array([0.2, 0.3, 0.5]), strength=1)
    assert result[0, 0] > 0.99
    np.testing.assert_allclose(result.sum(axis=1), 1)


def test_normalized_smoothing_shares_pretend_counts_by_the_prior_not_evenly():
    # one observation of the first code. the two unseen codes must not get equal chances:
    # the prior says the second is 7 times as likely as the third, so it gets 7 times the share
    result = mm.shrink([[1, 0, 0]], np.array([0.2, 0.7, 0.1]), strength=10)
    assert result[0, 1] / result[0, 2] == pytest.approx(7)
    np.testing.assert_allclose(result.sum(axis=1), 1)


def test_no_smoothing_gives_raw_proportions_with_epsilon_for_zeros():
    result = mm.shrink([[3, 1, 0]], np.array([0.2, 0.3, 0.5]), strength=0)
    np.testing.assert_array_equal(result[0], [0.75, 0.25, mm.EPSILON])


def test_no_smoothing_fills_an_empty_row_with_the_prior():
    prior = np.array([0.2, 0.3, 0.5])
    result = mm.shrink([[0, 0, 0], [2, 0, 0]], prior, strength=0)
    np.testing.assert_allclose(result[0], prior)
    np.testing.assert_array_equal(result[1], [1, mm.EPSILON, mm.EPSILON])


def test_smoothing_toward_own_frequencies_adds_nothing_to_a_code_never_used():
    # alice never used Approving anywhere, so even a strong pull leaves it at EPSILON.
    # Empty, which she used once elsewhere, gets a share from her own frequencies.
    rows = mm.add_previous_code(toy_reviews().iloc[[0, 2, 3]])  # Directing, Directing, Empty
    matrix = mm.personal_matrix(rows, CODES, strength=100)
    assert matrix.loc["Directing", "Approving"] == mm.EPSILON
    assert matrix.loc["Directing", "Empty"] > 0.1


def test_code_frequencies_refuse_a_missing_code():
    with pytest.raises(ValueError):
        mm.code_frequencies(pd.DataFrame({"Code": ["Approving", "Empty"]}), CODES)


def test_a_tie_goes_to_the_code_first_in_order():
    probs = pd.DataFrame([[0.5, 0.5, 0.0]], columns=CODES)
    assert mm.top_code(probs, ["Directing", "Approving", "Empty"]).iloc[0] == "Directing"


def test_cross_entropy_is_infinite_when_the_real_code_had_no_chance():
    probs = pd.DataFrame([[1.0, 0.0, 0.0]], columns=CODES)
    assert mm.log_loss_bits(probs, pd.Series(["Directing"])) == np.inf


def test_personal_matrix_has_every_state_and_rows_add_to_one():
    df = mm.add_previous_code(toy_reviews())
    matrix = mm.personal_matrix(df[df["commenter"] == "alice"], CODES, strength=1)

    assert list(matrix.index) == [mm.START] + CODES
    assert list(matrix.columns) == CODES
    np.testing.assert_allclose(matrix.sum(axis=1), 1)


def test_every_model_gives_probabilities_that_add_to_one():
    df = mm.add_previous_code(toy_reviews())
    train, rows = df.iloc[:4], df.iloc[4:]
    for probs in [
        mm.predict_overall_frequency(train, rows, CODES),
        mm.predict_personal_frequency(train, rows, CODES),
        mm.predict_overall_markov(train, rows, CODES, strength=1),
        mm.predict_personal_markov(train, rows, CODES, strength=1, population_strength=1),
    ]:
        assert list(probs.index) == list(rows.index)
        np.testing.assert_allclose(probs.sum(axis=1), 1)


def test_cross_entropy_of_a_uniform_guess():
    probs = pd.DataFrame(np.full((2, 4), 0.25), columns=["a", "b", "c", "d"])
    assert mm.log_loss_bits(probs, pd.Series(["a", "d"])) == pytest.approx(2.0)
