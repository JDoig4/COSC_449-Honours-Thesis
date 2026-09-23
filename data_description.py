"""print the corpus figures and preprocessing counts for the Deliverable 1 data description

run:  python data_description.py > data_description.txt
"""

import pandas as pd

from read_csv import dedupe_frame, load_reviews


# load the data twice: the raw frame exactly as it comes out of the CSVs, and
# the deduped frame the analysis actually runs on. keeping both means every
# figure below can be reported before and after preprocessing, which is what
# justifies the dedup decision in the report.
raw = load_reviews()
deduped = dedupe_frame(raw)


# --- 1. shape before and after dedup ---------------------------------------
# how much the dedup removed, plus the check that it removed only repeated
# (comment_id, Code) pairs and not whole comments: the number of distinct
# comment_id must be identical on both sides. if it is not, the dedup is
# deleting comments outright and the rule in dedupe_frame is wrong.
dropped = len(raw) - len(deduped)

print("=== shape ===")
print("raw rows:                  ", len(raw))
print("deduped rows:              ", len(deduped))
print("rows dropped:              ", dropped, "({:.1f}% of raw)".format(100 * dropped / len(raw)))
print("distinct comment_id, raw:  ", raw["comment_id"].nunique())
print("distinct comment_id, dedup:", deduped["comment_id"].nunique())
print()


# --- 2. what the corpus covers ---------------------------------------------
# headline figures for the data description. all are taken from the raw frame
# because dedup changes none of them.
#
# pr_id is counted as a (team, pr_id) pair, not on its own: PR numbers restart
# at 1 in every team's repository, so PR 15 in team 1 and PR 15 in team 3 are
# different pull requests. counting pr_id alone collapses them and undercounts
# roughly fourfold. commenter needs no such treatment - no commenter appears in
# more than one team, so the usernames are already globally distinct.
print("=== corpus ===")
print("teams:         ", raw["team"].nunique())
print("commenters:    ", raw["commenter"].nunique())
print("pull requests: ", len(raw.groupby(["team", "pr_id"])))
print("labels:        ", raw["Code"].nunique())
print("first comment: ", raw["created_at"].min())
print("last comment:  ", raw["created_at"].max())
print()


# --- 3. label distribution, before and after dedup --------------------------
# the class balance the baselines are working against, and the evidence that
# dedup was not a neutral operation: it removes a fifth of some classes and
# almost none of others, so the two frames give measurably different per-class
# metrics. pct_change is the deduped count as a percentage change from raw.
labels = pd.concat(
    [raw["Code"].value_counts(), deduped["Code"].value_counts()],
    axis=1,
    keys=["raw", "deduped"],
)
labels["pct_change"] = (100 * (labels["deduped"] / labels["raw"] - 1)).round(1)
labels["pct_of_deduped"] = (100 * labels["deduped"] / len(deduped)).round(1)

print("=== label distribution ===")
print(labels.to_string())
print()


# --- 4. duplicate rate per team --------------------------------------------
# how the duplication is spread across the 22 source files. this is the
# evidence for treating the duplicates as annotator behaviour rather than a
# systematic export fault: a bug in the export tooling would hit every file at
# a similar rate, whereas a rate that runs from 0% to roughly 24% points at the
# people doing the coding.
#
# the subset here must match the one in dedupe_frame, otherwise this table
# counts something different from what was actually dropped.
dup_mask = raw.duplicated(subset=["comment_id", "Code"])

teams = pd.DataFrame(
    {
        "rows": raw.groupby("team").size(),
        "dropped": raw[dup_mask].groupby("team").size(),
    }
).fillna({"dropped": 0})
teams["dropped"] = teams["dropped"].astype(int)
teams["pct"] = (100 * teams["dropped"] / teams["rows"]).round(1)

# team is stored as a string, so the default sort would order it 1, 10, 11, 2.
# sort numerically instead so the table reads in team order.
teams = teams.sort_index(key=lambda idx: idx.astype(int))

print("=== duplicate rate by team ===")
print(teams.to_string())
print()
print("teams with no duplicates at all:", (teams["dropped"] == 0).sum())
print("highest team duplicate rate:     {:.1f}%".format(teams["pct"].max()))
print("lowest team duplicate rate:      {:.1f}%".format(teams["pct"].min()))
