"""cluster commenters by their personal Markov models, and draw the clusters

each commenter's personal transition matrix (15 rows x 14 columns) is flattened into one row of
210 numbers, and k-means groups commenters whose matrices look alike. only commenters with enough
within-PR transitions are clustered, since a matrix built from a handful of pairs is mostly the
population matrix and says little about the person.

run from the project root:  .venv/Scripts/python.exe src/cluster_markov.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

import markov_models as mm
import read_csv
from markov_plots import CLUSTER_COLORS, CLUSTER_MARKERS, GRID, SECONDARY, plot_matrix, save

# a commenter needs at least this many within-PR transitions (START not counted) to be clustered
MIN_TRANSITIONS = 30
K_RANGE = range(2, 9)


def cluster(features, k):
    """k-means with many restarts; clusters are numbered 1, 2, ... from largest to smallest."""
    labels = KMeans(n_clusters=k, n_init=50, random_state=0).fit_predict(features)
    by_size = pd.Series(labels).value_counts().index
    renumber = {old: new for new, old in enumerate(by_size, start=1)}
    return np.array([renumber[label] for label in labels])


def plot_silhouette(scores, chosen):
    """Silhouette score for every k: higher means tighter, better separated clusters."""
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.plot(scores.index, scores.values, color=CLUSTER_COLORS[0], marker="o", markersize=5, linewidth=2)
    ax.scatter([chosen], [scores[chosen]], s=90, facecolors="none", edgecolors=SECONDARY, zorder=3)
    ax.set_xlabel("number of clusters (k)")
    ax.set_ylabel("silhouette")
    ax.set_title("Choosing k", loc="left")
    ax.grid(color=GRID, linewidth=0.8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    save(fig, "cluster_silhouette.png")


def plot_pca(features, labels, k):
    """Every clustered commenter as a dot, squashed from 210 numbers down to 2 with PCA."""
    pca = PCA(n_components=2, random_state=0)
    points = pca.fit_transform(features)
    share = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for c in range(1, k + 1):
        members = labels == c
        ax.scatter(points[members, 0], points[members, 1], s=40, color=CLUSTER_COLORS[c - 1],
                   marker=CLUSTER_MARKERS[c - 1], edgecolors="white", linewidths=0.8,
                   label=f"Cluster {c} ({members.sum()})", zorder=3)
    ax.set_xlabel(f"PC1 ({share[0]:.0%} of variance)")
    ax.set_ylabel(f"PC2 ({share[1]:.0%} of variance)")
    ax.set_title("Commenters' personal Markov models, coloured by cluster", loc="left")
    # lower left is empty in this plot; elsewhere the legend sits among the dots
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    save(fig, "cluster_pca.png")


def plot_cluster_differences(centres, population, sizes):
    """For each cluster: its average matrix minus the population matrix (red = more than usual)."""
    k = len(centres)
    limit = max(np.abs(c.values - population.values).max() for c in centres)
    fig, axes = plt.subplots(1, k, figsize=(5.4 * k, 5.4), sharey=True)
    for c, (ax, centre) in enumerate(zip(np.atleast_1d(axes), centres), start=1):
        plot_matrix(ax, centre - population, f"Cluster {c} ({sizes[c]} commenters) minus population",
                    diverging=True, vmax=limit, label_min=0.08)
        if c > 1:
            ax.set_ylabel("")
    fig.tight_layout()
    save(fig, "cluster_differences.png")


def plot_typical_members(matrices, typical):
    """The personal matrix of the commenter closest to each cluster's centre."""
    k = len(typical)
    fig, axes = plt.subplots(1, k, figsize=(5.4 * k, 5.4), sharey=True)
    for c, ax in enumerate(np.atleast_1d(axes), start=1):
        name = typical[c]
        plot_matrix(ax, matrices[name], f"Cluster {c}: most typical member ({name})", vmax=1)
        if c > 1:
            ax.set_ylabel("")
    fig.tight_layout()
    save(fig, "cluster_typical_members.png")


def describe(df, labels, eligible, centres, population):
    """One row per cluster: size, data, teams, code mix, and the transitions that stand out."""
    member_cluster = pd.Series(labels, index=eligible, name="cluster")

    # rows for rarely used previous codes mostly copy each member's own code mix (that is the
    # smoothing prior), so look for the transitions that stand out in START and the 4 common codes
    common_rows = [mm.START] + list(df["Code"].value_counts().index[:4])
    rows = []
    for c, centre in enumerate(centres, start=1):
        members = member_cluster[member_cluster == c].index
        utterances = df[df["commenter"].isin(members)]
        teams = df[df["commenter"].isin(members)].groupby("commenter")["team"].first()
        mix = utterances["Code"].value_counts(normalize=True)
        difference = (centre - population).loc[common_rows].stack().sort_values(ascending=False)
        rows.append({
            "cluster": c,
            "commenters": len(members),
            "median_utterances": int(utterances.groupby("commenter").size().median()),
            "teams": teams.nunique(),
            "largest_team_share": round(teams.value_counts(normalize=True).iloc[0], 2),
            "top_codes": ", ".join(f"{code} {share:.0%}" for code, share in mix.head(3).items()),
            "above_population": ", ".join(f"{a}->{b} +{d:.2f}" for (a, b), d in difference.head(3).items()),
            "below_population": ", ".join(f"{a}->{b} {d:.2f}" for (a, b), d in difference.tail(3).items()),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = mm.add_previous_code(read_csv.load_reviews())
    codes = sorted(df["Code"].unique())

    # descriptive models use all the data; strengths still picked by cross-validation
    population_strength, personal_strength, _ = mm.choose_markov_strengths(df, codes)
    population = mm.population_matrix(df, codes, population_strength)
    print(f"strengths: population {population_strength}, personal {personal_strength}")

    # who has enough within-PR transitions to cluster?
    transitions = df[df["previous"] != mm.START].groupby("commenter").size()
    transitions = transitions.reindex(df["commenter"].unique(), fill_value=0)
    eligible = transitions[transitions >= MIN_TRANSITIONS].index.sort_values()
    print(f"{len(eligible)} of {len(transitions)} commenters have >= {MIN_TRANSITIONS} within-PR transitions")

    # each eligible commenter's matrix, flattened to one row of 15 x 14 = 210 numbers
    matrices = {name: mm.personal_matrix(df[df["commenter"] == name], codes, personal_strength)
                for name in eligible}
    features = np.array([matrices[name].values.ravel() for name in eligible])

    # pick k with the highest silhouette
    silhouettes = pd.Series({k: silhouette_score(features, cluster(features, k)) for k in K_RANGE})
    k = int(silhouettes.idxmax())
    print("\nsilhouette by k:", silhouettes.round(3).to_dict(), f"-> k = {k}")
    labels = cluster(features, k)

    # is the result stable? compare with a different method (Ward hierarchical clustering).
    # adjusted Rand index: 1 = same groups, 0 = no better than chance.
    ward = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(features)
    print(f"agreement with Ward clustering (adjusted Rand index): {adjusted_rand_score(labels, ward):.2f}")

    # do the transitions matter, or is it just which codes people use? cluster the same people on
    # their code frequencies alone (14 numbers each) and compare the groups.
    frequencies = pd.crosstab(df["commenter"], df["Code"]).reindex(index=eligible, columns=codes)
    frequencies = frequencies.div(frequencies.sum(axis=1), axis=0).to_numpy()
    by_frequency = cluster(frequencies, k)
    print(f"agreement with clustering on code frequencies only (adjusted Rand index): "
          f"{adjusted_rand_score(labels, by_frequency):.2f}")

    # the same clustering with no smoothing: every row is its raw proportions, and a row with no
    # data takes the commenter's own code frequencies. do the groups survive without smoothing?
    raw = np.array([mm.personal_matrix(df[df["commenter"] == name], codes, 0).values.ravel()
                    for name in eligible])
    raw_silhouettes = pd.Series({j: silhouette_score(raw, cluster(raw, j)) for j in K_RANGE})
    print("\nno smoothing, silhouette by k:", raw_silhouettes.round(3).to_dict())
    print(f"agreement between no smoothing and normalized smoothing, both at k = {k} (adjusted Rand index): "
          f"{adjusted_rand_score(labels, cluster(raw, k)):.2f}")

    # each cluster's centre is the average of its members' matrices
    centres = [pd.DataFrame(features[labels == c].mean(axis=0).reshape(population.shape),
                            index=population.index, columns=codes) for c in range(1, k + 1)]
    sizes = {c: int((labels == c).sum()) for c in range(1, k + 1)}

    # the member whose matrix is closest to their cluster's centre
    typical = {}
    for c, centre in enumerate(centres, start=1):
        members = np.flatnonzero(labels == c)
        distances = np.linalg.norm(features[members] - centre.values.ravel(), axis=1)
        typical[c] = eligible[members[distances.argmin()]]

    summary = describe(df, labels, eligible, centres, population)
    print("\n=== clusters ===")
    print(summary.to_string(index=False))

    os.makedirs("outputs", exist_ok=True)
    summary.to_csv("outputs/cluster_summary.csv", index=False)
    pd.DataFrame({"commenter": eligible, "cluster": labels,
                  "transitions": transitions[eligible].values}).to_csv("outputs/clusters.csv", index=False)

    plot_silhouette(silhouettes, k)
    plot_pca(features, labels, k)
    plot_cluster_differences(centres, population, sizes)
    plot_typical_members(matrices, typical)
    print("\ntypical members:", typical)
    print("figures saved to report/figures/")
