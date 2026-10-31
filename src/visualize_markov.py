"""draw the population Markov model, fit on all the data: a heatmap and a state diagram

this is the descriptive model (no train/test split), so it uses every utterance.
the shrinkage strength is still picked by cross-validation, just on all the data.

run from the project root:  .venv/Scripts/python.exe src/visualize_markov.py
"""

import matplotlib.pyplot as plt

import markov_models as mm
import read_csv
from markov_plots import plot_matrix, plot_state_diagram, save

if __name__ == "__main__":
    df = mm.add_previous_code(read_csv.load_reviews())
    codes = sorted(df["Code"].unique())

    population_strength, _ = mm.choose_strength(df, codes, mm.predict_overall_markov)
    population = mm.population_matrix(df, codes, population_strength)
    print(f"population strength: {population_strength}")
    print(population.round(2).to_string())

    # the population model as a heatmap: each row adds up to 1
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    plot_matrix(ax, population, "Population Markov model: P(next code | previous code)")
    fig.tight_layout()
    save(fig, "population_matrix.png")

    # the same model as a state diagram, circles sized by how common each code is
    frequencies = df["Code"].value_counts(normalize=True)
    plot_state_diagram(population, frequencies, "population_state_diagram.png")
    print("\nfigures saved to report/figures/")
