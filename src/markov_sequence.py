import pandas as pd

import read_csv


def commenter_sequence(reviews, commenter):
    """Return one commenter's utterances in the order they happened."""

    # keep only the rows written by this person
    rows = reviews[reviews["commenter"] == commenter]

    # load_reviews() already sorts every row by created_at
    # rows with the exact same time (utterances from one comment) stay in their file order.
    return rows[["created_at", "pr_id", "comment_id", "Code"]]


def transition_counts(reviews):
    """Count how many transitions each commenter has under option A and option B.

    option A: one long sequence per person, so every utterance pairs with the next one.
    option B: cut the sequence every time the PR changes, so pairs only count inside one PR.
    """
    results = []
    for commenter, rows in reviews.groupby("commenter", sort=False):
        n = len(rows)

        # option A: n utterances in a row give n - 1 pairs
        option_a = n - 1

        # option B: count the pieces. a new piece starts whenever pr_id differs from the row above.
        # each piece of k utterances gives k - 1 pairs, so in total that's n - (number of pieces).
        pieces = (rows["pr_id"] != rows["pr_id"].shift()).sum()
        option_b = n - pieces

        results.append({"commenter": commenter, "utterances": n, "option_a": option_a, "option_b": option_b})

    return pd.DataFrame(results)


def count_matrix(rows, codes):
    """Count one commenter's transitions in a 14 x 14 table (option A).

    rows are one commenter's utterances in time order.
    each cell [from, to] = how many times this person did "from" and then "to" next.
    """
    # line every code up with the code that came right after it.
    # shift(-1) moves the Code column up by one row, so each row now also holds the next code.
    pairs = pd.DataFrame({"from": rows["Code"], "to": rows["Code"].shift(-1)})

    # the last utterance has no next code, so drop that one unfinished pair
    pairs = pairs.dropna()

    # count every (from, to) pair: rows are "from", columns are "to"
    matrix = pd.crosstab(pairs["from"], pairs["to"])

    # crosstab only includes codes this person used. put all 14 codes back, in the same order,
    # with 0 for pairs that never happened, so every commenter's table has the same shape.
    return matrix.reindex(index=codes, columns=codes, fill_value=0)


def all_count_matrices(reviews):
    """Build a count matrix for every commenter. Returns {commenter: matrix}."""

    # one fixed list of all 14 codes, so every matrix uses the same rows and columns
    codes = sorted(reviews["Code"].unique())

    return {commenter: count_matrix(rows, codes) for commenter, rows in reviews.groupby("commenter")}


if __name__ == "__main__":
    reviews = read_csv.load_reviews()

    # count how many utterances each commenter wrote
    counts = reviews["commenter"].value_counts()

    # pick the first commenter with 15 to 30 utterances
    # long enough to see pattern, short enough to read on screen
    commenter = counts[(counts >= 15) & (counts <= 30)].index[0]
    sequence = commenter_sequence(reviews, commenter)

    print(f"commenter: {commenter} ({len(sequence)} utterances)\n")
    print(sequence.to_string(index=False))

    # QUESTION: where should sequence be broken
    # a "PR change" is a row whose pr_id is different from the row above it.
    # we need to decide if "last code in one PR -> first code in the next PR" counts as a transition.
    pr_changes = (sequence["pr_id"] != sequence["pr_id"].shift()).sum() - 1
    print(f"\nnumber of times the PR changes: {pr_changes}")

    # QUESTION: ties.
    # some rows have same exact time, should i just sort by file order for those??
    tied = sequence["created_at"].duplicated(keep=False)
    print(f"rows that share a timestamp with another row: {tied.sum()}")

    #how many transitions does every commenter keep under option A vs option B?
    counts = transition_counts(reviews)
    print("\n=== transitions per commenter, all commenters ===")
    for option in ["option_a", "option_b"]:
        print(f"\n{option}:")
        print(f"  total transitions:              {counts[option].sum()}")
        print(f"  median per commenter:           {counts[option].median()}")
        print(f"  commenters with 0 transitions:  {(counts[option] == 0).sum()}")
        print(f"  commenters with under 10:       {(counts[option] < 10).sum()}")

    # step 2: count matrices
    matrices = all_count_matrices(reviews)

    # show the example commenter's table. hide all-zero rows and columns so it fits on screen.
    example = matrices[commenter]
    used = example.loc[(example.sum(axis=1) > 0), (example.sum(axis=0) > 0)]
    print(f"\n=== {commenter}'s transition counts (codes they never moved from/to are hidden) ===")
    print(used)

    # sanity check: under option A, every commenter's table should hold exactly n - 1 transitions
    n_utterances = reviews["commenter"].value_counts()
    all_match = all(matrix.values.sum() == n_utterances[name] - 1 for name, matrix in matrices.items())
    print(f"\nevery matrix is 14 x 14: {all(m.shape == (14, 14) for m in matrices.values())}")
    print(f"every matrix holds n - 1 transitions: {all_match}")
