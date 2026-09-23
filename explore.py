from read_csv import load_reviews

df = load_reviews()

# how many rows and unique comment IDs
print(len(df), df["comment_id"].nunique())
print(df["comment_id"].value_counts().head())


# which comment ID has the most rows and what are those rows
top_id = df["comment_id"].value_counts().index[0]
print(df[df["comment_id"] == top_id].to_string())

# how many comment IDs have multiple rows
# and what is the max number of unique teams/commenters/PRs for a single comment ID
multi = df["comment_id"].value_counts()
multi = multi[multi > 1].index
sub = df[df["comment_id"].isin(multi)]
print(sub.groupby("comment_id")[["team", "commenter", "pr_id"]].nunique().max())

# how many rows per comment ID, how many comment IDs have multiple rows
# and how many fully identical rows are there
sizes = df.groupby("comment_id").size()
print(sizes.describe())
print("comments with >1 utterance:", (sizes > 1).sum())
print("fully identical rows:", df.duplicated().sum())

