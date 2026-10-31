"""first-order Markov models of review behaviour, one per commenter

a sequence is one commenter's utterances inside one PR, in time order. every sequence starts in
a START state, so even a PR where someone wrote a single utterance tells us something: which
code they open with.

four models, each one adds one idea to the one before it:
  overall frequency    P(code)                      how common each code is, for everyone
  personal frequency   P(code | commenter)          how common each code is, for this person
  overall Markov       P(code | previous)           what usually comes next, for everyone
  personal Markov      P(code | previous, person)   what usually comes next, for this person

each model returns a table with one row per utterance and one column per code, holding the
chance the model gives that code. every row adds up to 1.

two smoothing options for the Markov models (see shrink):
  normalized   pretend observations are shared out in proportion to a prior, never evenly. for a
               personal matrix the prior is that commenter's own code frequencies, so the
               population plays no part and a code the person never used stays at (almost) 0.
               the strength is picked by cross-validation.
  none         strength 0: every row is its raw proportions

in both, a chance that would be exactly 0 is set to EPSILON instead, so the cross-entropy can be
computed. EPSILON is far too small to count as real probability.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

START = "START"

# the smoothing strengths tried when we pick one (see choose_strength)
STRENGTH_GRID = [0.1, 0.3, 1, 3, 10, 30, 100]

# the fixed stand-in for a chance of 0. log2(0) is undefined, log2(EPSILON) is about -33
EPSILON = 1e-10


def add_previous_code(df):
    """Add a 'previous' column: the code this commenter used just before, in the same PR.

    the first utterance a commenter makes in a PR gets START. utterances from one comment share a
    timestamp, so they keep their file order (load_reviews sorts with a stable sort).
    """
    # the order matters, so refuse a frame that isn't in time order
    if not df["created_at"].is_monotonic_increasing:
        raise ValueError("rows must be in time order (load_reviews() sorts them)")

    df = df.copy()

    # one sequence = one commenter inside one PR. pr_id numbers restart in every team,
    # so team is part of the key too.
    sequence = df.groupby(["team", "pr_id", "commenter"], sort=False)

    # shift(1) inside each sequence gives the row just before this one, or nothing for the first
    df["previous"] = sequence["Code"].shift(1).fillna(START)
    return df


def shrink(counts, prior, strength):
    """Turn counts into probabilities, pulled toward a prior when there is little data.

    probability = (count + strength * prior) / (row total + strength)

    a row with lots of data is mostly its own counts. strength says how many "pretend"
    observations the prior is worth, and they are shared out in proportion to the prior, so a
    code the prior says is rare gets only a small share.

    strength 0 means no smoothing: each row is just its raw proportions.
    a row with no data at all has nothing to go on, so it takes the prior whatever the strength.
    any chance still at exactly 0 (never seen, and the prior gives it nothing either) becomes
    EPSILON. that adds no real probability; it only keeps log2(0) out of the cross-entropy.
    counts and prior must have the same shape (prior can also be one row used for every row).
    """
    counts = np.asarray(counts, dtype=float)
    prior = np.broadcast_to(np.asarray(prior, dtype=float), counts.shape)
    totals = counts.sum(axis=1, keepdims=True)

    # with strength 0, an empty row would be 0 / 0, so silence that and use the prior instead
    with np.errstate(invalid="ignore"):
        probs = (counts + strength * prior) / (totals + strength)
    probs = np.where(totals > 0, probs, prior)
    return np.where(probs == 0, EPSILON, probs)


def code_frequencies(train, codes):
    """How often each code appears in train, as probabilities.

    no smoothing is needed here: every code appears in the training data. if one ever didn't,
    it would get a chance of 0 everywhere, so that is refused instead.
    """
    counts = train["Code"].value_counts().reindex(codes, fill_value=0)
    if (counts == 0).any():
        raise ValueError(f"codes missing from the training rows: {list(counts[counts == 0].index)}")
    return counts / counts.sum()


def transition_counts(df, codes):
    """Count (previous -> code) pairs in a 15 x 14 table: rows START + 14 codes, columns 14 codes."""
    states = [START] + list(codes)
    return pd.crosstab(df["previous"], df["Code"]).reindex(index=states, columns=codes, fill_value=0)


def population_matrix(train, codes, strength):
    """Everyone's transitions pooled into one transition matrix (15 x 14).

    each row is pulled toward the overall code frequencies, so rare rows (like after Apologizing)
    still give sensible chances.
    """
    counts = transition_counts(train, codes)
    probs = shrink(counts, code_frequencies(train, codes).values, strength)
    return pd.DataFrame(probs, index=counts.index, columns=codes)


def own_frequencies(rows, codes):
    """How often one commenter used each code (raw, so a code they never used is 0)."""
    counts = rows["Code"].value_counts().reindex(codes, fill_value=0)
    return counts / counts.sum()


def personal_matrix(rows, codes, strength):
    """One commenter's transition matrix (15 x 14), pulled toward their own code frequencies.

    rows are that commenter's utterances. a row this person has lots of data for is mostly their
    own transitions. a row they have never used is just their own code frequencies. the
    population plays no part, so a code this person never used stays at EPSILON.
    """
    counts = transition_counts(rows, codes)
    probs = shrink(counts, own_frequencies(rows, codes).values, strength)
    return pd.DataFrame(probs, index=counts.index, columns=codes)


def personal_matrices(train, codes, strength):
    """Every commenter's matrix, stacked into one table indexed by (commenter, previous)."""
    matrices = {
        commenter: personal_matrix(rows, codes, strength)
        for commenter, rows in train.groupby("commenter")
    }
    return pd.concat(matrices, names=["commenter", "previous"])


# --- the four models: learn from train, give probabilities for rows ----------------------------

def predict_overall_frequency(train, rows, codes, strength=None):
    """Everyone gets the same chances: how common each code is overall."""
    probs = np.tile(code_frequencies(train, codes).values, (len(rows), 1))
    return pd.DataFrame(probs, index=rows.index, columns=codes)


def predict_personal_frequency(train, rows, codes):
    """Each commenter's own code frequencies, never smoothed (a code they never used gets EPSILON)."""
    counts = pd.crosstab(train["commenter"], train["Code"]).reindex(columns=codes, fill_value=0)

    # look up each row's commenter. a commenter with no training rows has nothing personal to go
    # on, so shrink (with strength 0) gives them the overall frequencies. this never happens in
    # the test set; it can happen inside the cross-validation folds.
    counts = counts.reindex(rows["commenter"]).fillna(0)
    probs = shrink(counts, code_frequencies(train, codes).values, 0)
    return pd.DataFrame(probs, index=rows.index, columns=codes)


def predict_overall_markov(train, rows, codes, strength):
    """Chances of the next code given the previous one, pooled over everyone."""
    population = population_matrix(train, codes, strength)

    # each row's probabilities are the population row for its previous code
    return pd.DataFrame(population.loc[rows["previous"]].values, index=rows.index, columns=codes)


def predict_personal_markov(train, rows, codes, strength, population_strength):
    """Chances of the next code given the previous one, for this commenter."""
    stacked = personal_matrices(train, codes, strength)

    # look up each row's (commenter, previous) row in the stacked matrices
    keys = pd.MultiIndex.from_arrays([rows["commenter"], rows["previous"]])
    probs = stacked.reindex(keys)

    # a commenter with no training rows at all has no matrix and nothing personal to go on, so
    # use the population row. this never happens in the test set; it can in cross-validation folds
    population = population_matrix(train, codes, population_strength)
    missing = probs.isna().any(axis=1).to_numpy()
    probs.iloc[missing] = population.loc[rows["previous"].to_numpy()[missing]].values

    return pd.DataFrame(probs.values, index=rows.index, columns=codes)


# --- scoring and picking the smoothing strength --------------------------------------------------

def top_code(probs, order):
    """The code with the highest chance in each row. A tie goes to the code that comes first in order.

    with no smoothing, ties are common (a row seen twice, once each way), so pass the codes from
    most to least common in training: a tie then goes to the more common code.
    """
    return probs[list(order)].idxmax(axis=1)


def chance_of_actual(probs, actual):
    """The chance each row's model gave the code that really happened."""
    return probs.to_numpy()[np.arange(len(actual)), probs.columns.get_indexer(actual)]


def log_loss_bits(probs, actual):
    """Cross-entropy in bits: the average of -log2(chance the model gave the code that happened).

    lower is better. a model that guesses uniformly over 14 codes scores log2(14) = 3.81 bits.
    the models never give exactly 0 (see EPSILON), but a code that happened after being given
    EPSILON costs about 33 bits on its own. a chance of exactly 0 would make the answer infinite.
    """
    with np.errstate(divide="ignore"):
        return float(-np.log2(chance_of_actual(probs, actual)).mean())


def choose_strength(train, codes, predict, grid=STRENGTH_GRID, n_splits=5, **fixed):
    """Pick the smoothing strength with the lowest cross-entropy, using only the training rows.

    the training rows are split into n_splits folds, grouped by comment_id like the main split.
    each strength is fit on all but one fold and scored on the one left out.
    returns the best strength and a Series of the average score for every strength.
    """
    folds = list(GroupKFold(n_splits=n_splits).split(train, groups=train["comment_id"]))

    scores = {}
    for strength in grid:
        fold_scores = []
        for fit_idx, score_idx in folds:
            fit_rows, score_rows = train.iloc[fit_idx], train.iloc[score_idx]
            probs = predict(fit_rows, score_rows, codes, strength, **fixed)
            fold_scores.append(log_loss_bits(probs, score_rows["Code"]))
        scores[strength] = np.mean(fold_scores)

    scores = pd.Series(scores, name="cross_entropy_bits")
    return scores.idxmin(), scores


def choose_markov_strengths(train, codes):
    """Pick both Markov strengths: first the population one, then the personal one on top of it.

    returns (population_strength, personal_strength, table of scores for every strength tried).
    """
    population_strength, population_scores = choose_strength(train, codes, predict_overall_markov)
    personal_strength, personal_scores = choose_strength(
        train, codes, predict_personal_markov, population_strength=population_strength)

    scores = pd.DataFrame({"overall_markov": population_scores, "personal_markov": personal_scores})
    return population_strength, personal_strength, scores
