"""drawing helpers shared by the Markov scripts: colours, transition-matrix heatmaps, state diagram"""

import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from markov_models import START

# figures for the report go here (outputs/ is not committed, this folder is)
FIGURES = "report/figures"

# ink and chart chrome
INK, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

# one fixed colour (and marker) per model, used in every results chart.
# the D1 overall baseline is the grey reference line the others are compared with.
MODEL_STYLE = {
    "overall_majority":   {"label": "Overall majority (D1)",  "color": MUTED,     "marker": "s"},
    "personal_majority":  {"label": "Personal majority (D1)", "color": "#1baf7a", "marker": "^"},
    "overall_frequency":  {"label": "Overall frequency",      "color": MUTED,     "marker": "s"},
    "personal_frequency": {"label": "Personal frequency",     "color": "#1baf7a", "marker": "^"},
    "overall_markov":     {"label": "Overall Markov",         "color": "#eb6834", "marker": "D"},
    "personal_markov":    {"label": "Personal Markov",        "color": "#2a78d6", "marker": "o"},
}

# cluster colours, in a fixed order (cluster 1 is always blue)
CLUSTER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
CLUSTER_MARKERS = ["o", "s", "^", "D", "v", "P"]

# sequential (one hue, light to dark) for probabilities, diverging (blue - grey - red) for differences
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blues", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_red", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f2a3a3", "#e34948", "#a32a2a"])

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 9,
    "text.color": INK,
    "axes.labelcolor": SECONDARY,
    "axes.edgecolor": AXIS,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "xtick.color": SECONDARY,
    "ytick.color": SECONDARY,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def save(fig, name):
    """Save a figure into the report's figures folder and close it."""
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(os.path.join(FIGURES, name))
    plt.close(fig)


def short(value):
    """0.36 -> '.36', so numbers fit inside heatmap cells."""
    return f"{value:.2f}".replace("0.", ".", 1)


def plot_matrix(ax, matrix, title, diverging=False, vmax=None, label_min=0.10):
    """Draw one transition matrix as a heatmap: rows = previous code, columns = next code.

    only cells of at least label_min (in size) get a number, so the strong transitions stand out.
    diverging=True is for differences (e.g. a cluster minus the population): blue below, red above.
    """
    values = matrix.to_numpy()
    if diverging:
        limit = vmax or np.abs(values).max()
        image = ax.imshow(values, cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
    else:
        limit = vmax or values.max()
        image = ax.imshow(values, cmap=SEQUENTIAL, vmin=0, vmax=limit, aspect="auto")

    # write the number in the strong cells, in white on dark cells so it stays readable
    for i, j in zip(*np.nonzero(np.abs(values) >= label_min)):
        dark = abs(values[i, j]) > 0.6 * limit
        ax.text(j, i, short(values[i, j]), ha="center", va="center", fontsize=6.5,
                color="white" if dark else INK)

    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=50, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(matrix.index)), matrix.index, fontsize=7.5)
    ax.set_xlabel("next code")
    ax.set_ylabel("previous code")
    ax.set_title(title, loc="left")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def plot_state_diagram(matrix, frequencies, name, threshold=0.2):
    """Draw the population Markov model as a graph of states and arrows.

    each code is a circle, sized by how common it is. an arrow A -> B is drawn when
    P(B | A) >= threshold, thicker for higher chances. staying on the same code is written under
    the code's name (the loop arrow would hide the label).

    almost every arrow points at the three most common codes, so those three sit in the middle and
    START plus the other codes sit on an outer ring. that way the arrows stay short and don't cross.
    """
    codes = list(matrix.columns)
    order = list(frequencies.sort_values(ascending=False).index)
    hubs, others = order[:3], [START] + order[3:]

    def ring(nodes, radius, start_angle):
        angles = start_angle - 2 * np.pi * np.arange(len(nodes)) / len(nodes)
        return {node: (radius * np.cos(a), radius * np.sin(a)) for node, a in zip(nodes, angles)}

    positions = {**ring(hubs, 0.36, 5 * np.pi / 6), **ring(others, 1.0, np.pi / 2)}

    graph = nx.DiGraph()
    graph.add_nodes_from([START] + codes)
    for previous in matrix.index:
        for following in codes:
            chance = matrix.loc[previous, following]
            if previous != following and chance >= threshold:
                graph.add_edge(previous, following, weight=chance)

    fig, ax = plt.subplots(figsize=(8.2, 8.2))

    # circles: area grows with how common the code is; START is a small grey circle
    sizes = [400 if node == START else 300 + 9000 * frequencies[node] for node in graph.nodes]
    colors = [AXIS if node == START else "#cde2fb" for node in graph.nodes]
    nx.draw_networkx_nodes(graph, positions, node_size=sizes, node_color=colors,
                           edgecolors=[MUTED if n == START else "#2a78d6" for n in graph.nodes],
                           linewidths=1.2, ax=ax)

    # arrows: width and darkness grow with the chance
    edges = list(graph.edges(data="weight"))
    nx.draw_networkx_edges(
        graph, positions, edgelist=[(a, b) for a, b, _ in edges],
        width=[0.8 + 7 * w for _, _, w in edges],
        edge_color=[w for _, _, w in edges], edge_cmap=SEQUENTIAL, edge_vmin=0, edge_vmax=0.45,
        arrows=True, arrowstyle="-|>", arrowsize=12, node_size=sizes,
        connectionstyle="arc3,rad=0.12", ax=ax,
    )
    nx.draw_networkx_edge_labels(
        graph, positions, edge_labels={(a, b): short(w) for a, b, w in edges},
        font_size=7, font_color=SECONDARY, connectionstyle="arc3,rad=0.12",
        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85}, ax=ax,
    )

    # code names (with the chance of staying on the same code): inside the big middle circles,
    # outside the circle for the ring
    for code in codes:
        x, y = positions[code]
        label = f"{code}\nstays {short(matrix.loc[code, code])}"
        if code in hubs:
            ax.text(x, y, label, ha="center", va="center", fontsize=8.5, color=INK, fontweight="bold")
        else:
            ax.text(1.22 * x, 1.22 * y, label, ha="center", va="center", fontsize=8.5, color=INK)
    x, y = positions[START]
    ax.text(1.17 * x, 1.17 * y, "START\n(first code in a PR)", ha="center", va="center",
            fontsize=8.5, color=SECONDARY)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Population Markov model: transitions with chance ≥ {threshold:.1f}", loc="left")
    save(fig, name)
