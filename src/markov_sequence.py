import read_csv


def commenter_sequence(reviews, commenter):
    """Return one commenter's utterances in the order they happened."""

    # keep only the rows written by this person
    rows = reviews[reviews["commenter"] == commenter]

    # load_reviews() already sorts every row by created_at
    # rows with the exact same time (utterances from one comment) stay in their file order.
    return rows[["created_at", "pr_id", "comment_id", "Code"]]


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
