# Draft: Personalized First-Order Markov Models

## 1. What the model is

The 14 different codes are the states. Each commenter will get their own matrix that shows the chance of a code following each other code. It is first-order, meaning the next code depends only on the current code.

## 2. How a sequence is defined

Each commenter's utterances are sorted into one long sequence by created_at. I considered breaking them up by PR, but this would leave too many commenters with data sets too small to build matrices with.

Example (bcalderon's first few utterances):

```
Approving (pr 28) --> Approving (pr 29) --> Approving (pr 27) -->
Directing (pr 27) --> Approving (pr 33) --> Approving (pr 33) --> ...
```

**Why not cut the sequence at each PR (option B)?**

```
option_a:
  total transitions:              16166
  median per commenter:           72.0
  commenters with 0 transitions:  0
  commenters with under 10:       3

option_b:
  total transitions:              9993
  median per commenter:           33.0
  commenters with 0 transitions:  3
  commenters with under 10:       24
```

Option B leaves too many commenters with < 10 transitions.

**Ties:**
The current idea for comments that are tied for timestamps is to keep them in file (row) order, but I am open to suggestions.

## 3. Variables

| Variable | What it is | Why it's useful |
|---|---|---|
| from / to | The code of an utterance, and the code of the same commenter's next utterance | This pair is the transition the matrix counts |
| commenter | Who wrote both utterances | Each commenter gets their own matrix (personalized) |
| same_comment | Whether both utterances are in the same comment | Shows if a pair is one message or two separate ones |
| same_pr | Whether both utterances are in the same PR | Separates pairs inside one review from pairs that jump to a different PR |
| gap_hours | Hours between the two utterances (0 inside one comment) | A pair minutes apart is more likely to be related than one weeks apart |
| week | Week of the term the pair happened in (0 to 27) | Lets me compare behaviour early vs late in the term |

32% of transitions are inside one comment and 62% are inside one PR. When the PR changes, the
median gap is 9.8 hours, but some gaps are over 50 days.

## 4. Building the transition matrix

1. Count each pair (code and next immediate code from sequence) into a 14x14 table
2. Divide each row by its total
3. Leave rows with no data empty

**Example (bcalderon):**

Counts for Approving and Criticizing

```
to           Agreeing  Approving  Criticizing  Directing  Empty  Updating
Approving           1         15            1          1      1         2
Criticizing         0          1            0          0      0         0
```

Probabilities for Approving and Criticizing

```
to           Agreeing  Approving  Criticizing  Directing  Empty  Updating
Approving        0.05       0.71         0.05       0.05   0.05       0.1
Criticizing      0.00       1.00         0.00       0.00   0.00       0.0
```

## 5. Questions for feedback

1. **Sparse data:** Empty rows per commenter (out of 14): median 6, min 0, max 13. For empty rows, should I leave them empty, smooth them by adding 1s, or fill them from a matrix made of overall behaviours?

2. **Long gaps:** Some commenters have very large gaps between utterances. For example, crystalalexander has an Approving --> Complimenting transition that is 1274.74 hours (about 53 days) apart. Should I weigh these transitions down, drop them entirely or just keep them?

3. **Minimum utterances:** Is there a minimum number of utterances per commenter that I should set a cutoff at?
