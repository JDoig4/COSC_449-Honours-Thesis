Initial note after merging CSVs:

There are some comments that have multiple codes, so I'll say that one instance is one code and one comment ID.

Duplicate rows are purposeful. No need to dedupe.

Since I plan to train/test split the data, I also need to group by comment id so that I dont have leakage.


Empty label is a predictor, not a mistake.
