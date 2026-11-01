"""Q3(b): effect of tree depth on bias^2 and variance, single tree vs 100-tree bagging."""
from common import setup, finish, load_data_and_split, load_artifact, save_artifact, save_table

display = setup("q3b_depth_bias_variance")

# ============================================================
# Q3(b): EFFECT OF TREE DEPTH ON BIAS AND VARIANCE
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from q3_helpers import (
    classes_q3, q3_bootstrap_indices, q3_single_tree_vector,
    q3_bagging_vector, q3_bias_variance,
)

X, y, X_train, y_train, X_val, y_val, X_test, y_test, seed = load_data_and_split()

best_tree = load_artifact("q1a_best_tree", produced_by="q1a_tree_grid")["best_tree"]

# Same line as in the notebook's Q3(a) cell.
selected_tree_leaf = best_tree.get_params()["min_samples_leaf"]

# ---------------- original notebook code (unchanged) ----------------
R_DEPTH = 30
B_DEPTH = 100

# Use the selected tree's actual fitted depth.
selected_depth_q3 = best_tree.get_depth()

depth_settings_q3 = [
    ("Shallow", 2),
    ("Selected", selected_depth_q3),
    ("Deep", max(32, selected_depth_q3 + 5))
]

# Require three distinct depth settings.
depth_values = [depth for _, depth in depth_settings_q3]

if len(set(depth_values)) != 3:
    raise ValueError(
        "The three depths are not distinct. Adjust the shallow "
        "depth in depth_settings_q3."
    )

rng_depth_q3 = np.random.default_rng(seed + 5000)

depth_bv_rows = []
depth_predictions = {}
depth_accuracy_records = {}

for depth_name, depth in depth_settings_q3:

    print(f"\nDepth setting: {depth_name} (max_depth={depth})")

    single_vectors = []
    bag_vectors = []

    single_accuracies = []
    bag_accuracies = []

    for rep in range(R_DEPTH):

        # Six faces sampled with replacement within each identity.
        outer_idx = q3_bootstrap_indices(y_train, rng_depth_q3)
        X_boot = X_train[outer_idx]
        y_boot = y_train[outer_idx]

        # Single decision tree.
        single_tree = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_leaf=selected_tree_leaf,
            random_state=seed + 60000 + rep
        )
        single_tree.fit(X_boot, y_boot)

        p_single = q3_single_tree_vector(single_tree, X_test)
        single_vectors.append(p_single)

        single_accuracies.append(
            accuracy_score(
                y_test, classes_q3[np.argmax(p_single, axis=1)]
            )
        )

        # 100-tree bagging ensemble.
        bagged_trees = []

        for b in range(B_DEPTH):

            inner_rng = np.random.default_rng(
                seed + 70000 + rep * B_DEPTH + b
            )

            inner_idx = inner_rng.choice(
                len(X_boot), size=len(X_boot), replace=True
            )

            bag_tree = DecisionTreeClassifier(
                max_depth=depth,
                min_samples_leaf=selected_tree_leaf,
                random_state=seed + 80000 + rep * B_DEPTH + b
            )

            bag_tree.fit(X_boot[inner_idx], y_boot[inner_idx])
            bagged_trees.append(bag_tree)

        p_bag = q3_bagging_vector(bagged_trees, X_test)
        bag_vectors.append(p_bag)

        bag_accuracies.append(
            accuracy_score(
                y_test, classes_q3[np.argmax(p_bag, axis=1)]
            )
        )

        if (rep + 1) % 10 == 0:
            print(f"  Completed {rep + 1}/{R_DEPTH} repetitions")

    single_vectors = np.asarray(single_vectors)
    bag_vectors = np.asarray(bag_vectors)

    # Measured bias squared and variance.
    single_bias2, single_variance = q3_bias_variance(
        single_vectors, y_test
    )

    bag_bias2, bag_variance = q3_bias_variance(
        bag_vectors, y_test
    )

    single_mean_acc = np.mean(single_accuracies)
    bag_mean_acc = np.mean(bag_accuracies)

    # Store measurements for Q3(c).
    depth_predictions[(depth_name, "Single tree")] = single_vectors
    depth_predictions[(depth_name, "Bagging")] = bag_vectors

    depth_accuracy_records[(depth_name, "Single tree")] = np.asarray(
        single_accuracies
    )
    depth_accuracy_records[(depth_name, "Bagging")] = np.asarray(
        bag_accuracies
    )

    # Store one row for each model at this depth.
    for model_name, bias2, variance, accuracies in [
        ("Single tree", single_bias2, single_variance,
         single_accuracies),
        ("100-tree bagging", bag_bias2, bag_variance,
         bag_accuracies)
    ]:
        depth_bv_rows.append({
            "Depth setting": depth_name,
            "Max depth": depth,
            "Model": model_name,
            "Bias squared": bias2,
            "Variance": variance,
            "Mean test accuracy": np.mean(accuracies),
            "Accuracy SD": np.std(accuracies, ddof=1)
        })


depth_bv_results = pd.DataFrame(depth_bv_rows)

print("\nQ3(b): Bias, variance and accuracy at each depth")
display(depth_bv_results.round(5))


# ------------------------------------------------------------
# Quantify variance removed by bagging at each depth.
# ------------------------------------------------------------

variance_rows = []

for depth_name, depth in depth_settings_q3:

    single_row = depth_bv_results[
        (depth_bv_results["Depth setting"] == depth_name) &
        (depth_bv_results["Model"] == "Single tree")
    ].iloc[0]

    bag_row = depth_bv_results[
        (depth_bv_results["Depth setting"] == depth_name) &
        (depth_bv_results["Model"] == "100-tree bagging")
    ].iloc[0]

    variance_removed = (
        single_row["Variance"] - bag_row["Variance"]
    )

    variance_reduction_pct = (
        100 * variance_removed / single_row["Variance"]
        if single_row["Variance"] > 0 else np.nan
    )

    variance_rows.append({
        "Depth setting": depth_name,
        "Max depth": depth,
        "Single-tree variance": single_row["Variance"],
        "Bagging variance": bag_row["Variance"],
        "Variance removed": variance_removed,
        "Variance reduction (%)": variance_reduction_pct,
        "Single-tree accuracy": single_row["Mean test accuracy"],
        "Bagging accuracy": bag_row["Mean test accuracy"],
        "Accuracy improvement": (
            bag_row["Mean test accuracy"] -
            single_row["Mean test accuracy"]
        )
    })


depth_variance_comparison = pd.DataFrame(variance_rows)

print("\nVariance reduction and accuracy improvement by depth")
display(depth_variance_comparison.round(5))


# ------------------------------------------------------------
# Plot bias squared and variance versus depth.
# ------------------------------------------------------------

for metric in ["Bias squared", "Variance"]:

    plt.figure(figsize=(8, 5))

    for model_name in ["Single tree", "100-tree bagging"]:

        subset = depth_bv_results[
            depth_bv_results["Model"] == model_name
        ]

        plt.plot(
            subset["Depth setting"],
            subset[metric],
            marker="o",
            label=model_name
        )

    plt.xlabel("Tree depth setting")
    plt.ylabel(metric)
    plt.title(f"Q3(b): {metric} versus tree depth")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# Plot test accuracy versus depth.
plt.figure(figsize=(8, 5))

for model_name in ["Single tree", "100-tree bagging"]:

    subset = depth_bv_results[
        depth_bv_results["Model"] == model_name
    ]

    plt.plot(
        subset["Depth setting"],
        subset["Mean test accuracy"],
        marker="o",
        label=model_name
    )

plt.xlabel("Tree depth setting")
plt.ylabel("Mean clean test accuracy")
plt.title("Q3(b): Test accuracy versus tree depth")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# --------------------------------------------------------------------

save_table(depth_bv_results, "q3b_depth_bias_variance")
save_table(depth_variance_comparison, "q3b_variance_reduction")
save_artifact("q3b_depth_results", {
    "R_DEPTH": R_DEPTH,
    "B_DEPTH": B_DEPTH,
    "depth_settings_q3": depth_settings_q3,
    "depth_bv_results": depth_bv_results,
    "depth_variance_comparison": depth_variance_comparison,
})
finish()
