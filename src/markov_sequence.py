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


def probability_matrix(counts):
    """Turn a count matrix into a transition matrix of probabilities.

    each row is divided by its own total, so each row sums to 1.
    cell [from, to] then reads: "after doing 'from', the chance that 'to' comes next".

    a row for a code the person never moved from has a total of 0. dividing by 0 gives NaN
    ("no data") for that whole row. we leave it as NaN on purpose (option 1): it is honest about
    the missing data, and how to fill it is a question for later.
    """
    row_totals = counts.sum(axis=1)

    # div(..., axis=0) divides every cell by the total of its own row
    return counts.div(row_totals, axis=0)


def transitions_table(reviews):
    """Make one row per transition (option A), with simple variables about time.

    each row is one pair: a commenter did "from", and then "to" was the next thing they did.
    the extra columns describe that pair:
      same_comment: both utterances are in the same comment
      same_pr:      both utterances are in the same PR
      gap_hours:    hours between the two utterances (0 inside one comment)
      week:         week of the term the pair started in (week 0 = week of the first comment in the data)
    """
    # group by commenter so the "next" utterance is always the same person's next one.
    # reviews is already in time order, and groupby keeps that order inside each person.
    by_person = reviews.groupby("commenter")

    # shift(-1) inside each person's group gives the row right after this one
    table = pd.DataFrame({
        "commenter": reviews["commenter"],
        "from": reviews["Code"],
        "to": by_person["Code"].shift(-1),
        "same_comment": reviews["comment_id"] == by_person["comment_id"].shift(-1),
        "same_pr": reviews["pr_id"] == by_person["pr_id"].shift(-1),
        "gap_hours": (by_person["created_at"].shift(-1) - reviews["created_at"]).dt.total_seconds() / 3600,
        "week": (reviews["created_at"] - reviews["created_at"].min()).dt.days // 7,
    })

    # each person's last utterance has no next one, so it is not a transition
    return table.dropna(subset=["to"]).reset_index(drop=True)


def rows_are_valid(probabilities):
    """Check a transition matrix: each row is either full and sums to 1, or completely empty (NaN)."""

    full = probabilities.notna().all(axis=1)
    empty = probabilities.isna().all(axis=1)

    # no row should be half filled in
    if not (full | empty).all():
        return False

    # full rows should add up to 1 (allowing for tiny rounding errors)
    return ((probabilities[full].sum(axis=1) - 1).abs() < 1e-9).all()


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

    # step 3: transition (probability) matrices
    probabilities = {name: probability_matrix(matrix) for name, matrix in matrices.items()}

    # show the example commenter's probabilities for the same codes as above, rounded to read easily
    example_p = probabilities[commenter]
    print(f"\n=== {commenter}'s transition probabilities (same codes as above) ===")
    print(example_p.loc[used.index, used.columns].round(2))

    # sanity check: every row with data should add up to 1, and every other row should be all NaN
    rows_ok = all(rows_are_valid(p) for p in probabilities.values())
    print(f"\nevery row sums to 1 or is all NaN: {rows_ok}")

    # how much is missing? count the empty (NaN) rows each commenter has, out of 14
    empty_rows = pd.Series({name: p.isna().all(axis=1).sum() for name, p in probabilities.items()})
    print(f"empty rows per commenter (out of 14): median {empty_rows.median()}, min {empty_rows.min()}, max {empty_rows.max()}")

    # step 4: one row per transition, with the "over time" variables
    transitions = transitions_table(reviews)
    print("\n=== transitions table (first 8 rows) ===")
    print(transitions.head(8).round(2).to_string(index=False))

    # sanity check: it should hold the same 16166 transitions as option A
    print(f"\ntransitions in table: {len(transitions)} (option A total: {counts['option_a'].sum()})")

    # quick look at what the time variables show across everyone
    print(f"inside the same comment: {transitions['same_comment'].mean():.0%}")
    print(f"inside the same PR:      {transitions['same_pr'].mean():.0%}")
    print(f"median gap between utterances (hours), when the PR changes: {transitions.loc[~transitions['same_pr'], 'gap_hours'].median():.1f}")
    print(f"weeks covered: {transitions['week'].min()} to {transitions['week'].max()}")
