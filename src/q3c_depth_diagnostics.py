"""Q3(c): leaf occupancy and tree disagreement vs depth; where does bagging help most?"""
from common import setup, finish, load_data_and_split, load_artifact, save_table

display = setup("q3c_depth_diagnostics")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from q3_helpers import q3_bootstrap_indices

X, y, X_train, y_train, X_val, y_val, X_test, y_test, seed = load_data_and_split()

best_tree = load_artifact("q1a_best_tree", produced_by="q1a_tree_grid")["best_tree"]
selected_tree_leaf = best_tree.get_params()["min_samples_leaf"]

_q3b = load_artifact("q3b_depth_results", produced_by="q3b_depth_bias_variance")
R_DEPTH = _q3b["R_DEPTH"]
B_DEPTH = _q3b["B_DEPTH"]
depth_settings_q3 = _q3b["depth_settings_q3"]
depth_variance_comparison = _q3b["depth_variance_comparison"]

# ---------------- original notebook code (unchanged) ----------------
# ============================================================
# Q3(c): DEPTH WITH THE LARGEST BAGGING ACCURACY IMPROVEMENT
# Leaf occupancy and tree disagreement measurements
# ============================================================

from itertools import combinations

rng_diag_q3 = np.random.default_rng(seed + 9000)

diagnostic_rows = []

for depth_name, depth in depth_settings_q3:

    print(f"\nDiagnostics for {depth_name} depth: {depth}")

    single_leaf_occupancies = []
    bag_leaf_occupancies = []
    disagreements = []

    for rep in range(R_DEPTH):

        # Recreate a stratified outer bootstrap sample.
        outer_idx = q3_bootstrap_indices(y_train, rng_diag_q3)
        X_boot = X_train[outer_idx]
        y_boot = y_train[outer_idx]

        # Single tree: leaf occupancy.
        single_tree = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_leaf=selected_tree_leaf,
            random_state=seed + 100000 + rep
        )
        single_tree.fit(X_boot, y_boot)

        _, counts = np.unique(
            single_tree.apply(X_boot), return_counts=True
        )

        single_leaf_occupancies.append(np.mean(counts))

        # Construct 100 trees for occupancy and disagreement.
        trees = []
        predictions = []

        for b in range(B_DEPTH):

            inner_rng = np.random.default_rng(
                seed + 110000 + rep * B_DEPTH + b
            )

            inner_idx = inner_rng.choice(
                len(X_boot), size=len(X_boot), replace=True
            )

            tree = DecisionTreeClassifier(
                max_depth=depth,
                min_samples_leaf=selected_tree_leaf,
                random_state=seed + 120000 + rep * B_DEPTH + b
            )

            tree.fit(X_boot[inner_idx], y_boot[inner_idx])
            trees.append(tree)

            # Occupancy measured on the current outer training set.
            _, tree_counts = np.unique(
                tree.apply(X_boot), return_counts=True
            )
            bag_leaf_occupancies.append(np.mean(tree_counts))

            predictions.append(tree.predict(X_test))

        predictions = np.asarray(predictions)

        # Pairwise disagreement:
        # fraction of test examples on which two trees disagree.
        pair_disagreements = []

        for a, b in combinations(range(B_DEPTH), 2):
            pair_disagreements.append(
                np.mean(predictions[a] != predictions[b])
            )

        disagreements.append(np.mean(pair_disagreements))

        if (rep + 1) % 10 == 0:
            print(f"  Completed {rep + 1}/{R_DEPTH} repetitions")


    # Accuracy improvement and variance reduction come from Q3(b).
    result_row = depth_variance_comparison[
        depth_variance_comparison["Depth setting"] == depth_name
    ].iloc[0]

    diagnostic_rows.append({
        "Depth setting": depth_name,
        "Max depth": depth,
        "Mean single-tree leaf occupancy": np.mean(
            single_leaf_occupancies
        ),
        "Mean bagged-tree leaf occupancy": np.mean(
            bag_leaf_occupancies
        ),
        "Mean pairwise disagreement": np.mean(disagreements),
        "Single-tree accuracy": result_row["Single-tree accuracy"],
        "Bagging accuracy": result_row["Bagging accuracy"],
        "Accuracy improvement": result_row["Accuracy improvement"],
        "Variance removed": result_row["Variance removed"],
        "Variance reduction (%)": result_row["Variance reduction (%)"]
    })


depth_diagnostics = pd.DataFrame(diagnostic_rows)

print("\nQ3(c): Leaf occupancy, disagreement and bagging benefit")
display(depth_diagnostics.round(5))


# ------------------------------------------------------------
# Answer the key question from the measured results.
# ------------------------------------------------------------

best_improvement_row = depth_diagnostics.loc[
    depth_diagnostics["Accuracy improvement"].idxmax()
]

deepest_row = depth_diagnostics.loc[
    depth_diagnostics["Depth setting"] == "Deep"
].iloc[0]

print("\nQ3(c) measured conclusion")

print(
    "Largest accuracy improvement from bagging:",
    best_improvement_row["Depth setting"],
    f"({best_improvement_row['Accuracy improvement']:.5f})"
)

print(
    "Deep-tree accuracy improvement:",
    f"{deepest_row['Accuracy improvement']:.5f}"
)

if best_improvement_row["Depth setting"] == "Deep":
    print(
        "In this experiment, the deepest tree gave the largest "
        "measured accuracy improvement. This is an empirical result, "
        "not a guarantee that the deepest tree always benefits most."
    )
else:
    print(
        "In this experiment, the deepest tree did NOT give the largest "
        "measured accuracy improvement. The results show that depth "
        "alone does not determine the benefit of bagging."
    )


# Show the diagnostics in a compact comparison plot.
plt.figure(figsize=(8, 5))

plt.plot(
    depth_diagnostics["Depth setting"],
    depth_diagnostics["Mean pairwise disagreement"],
    marker="o",
    label="Pairwise disagreement"
)

plt.xlabel("Tree depth setting")
plt.ylabel("Mean disagreement")
plt.title("Q3(c): Tree disagreement versus depth")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# --------------------------------------------------------------------

save_table(depth_diagnostics, "q3c_depth_diagnostics")
finish()
