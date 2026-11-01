"""Q3(a): measured bias^2, variance and accuracy stability of tree / bagging / boosting."""
from common import setup, finish, load_data_and_split, load_artifact, save_table

display = setup("q3a_bias_variance")

# ============================================================
# Q3(a): MEASURED BIAS^2, VARIANCE AND ACCURACY STABILITY
# Consistent with the variables in your existing notebook
# ============================================================

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from samme import SAMMEBoosting
from q3_helpers import (
    classes_q3, n_classes_q3, class_to_col_q3,
    q3_one_hot, q3_bootstrap_indices, q3_align_proba,
    q3_single_tree_vector, q3_bagging_vector, q3_boosting_vector,
    q3_bias_variance,
)

X, y, X_train, y_train, X_val, y_val, X_test, y_test, seed = load_data_and_split()

best_tree = load_artifact("q1a_best_tree", produced_by="q1a_tree_grid")["best_tree"]
final_trees = load_artifact("q1d_final_trees", produced_by="q1d_masked_robustness")["final_trees"]
best_boost_model = load_artifact("q2b_best_boost_model", produced_by="q2b_samme_grid")["best_boost_model"]

# ---------------- original notebook code (unchanged) ----------------
# Check that the original model-selection cells have been run.
required_variables = [
    "X_train", "y_train", "X_test", "y_test",
    "best_tree", "best_boost_model", "final_trees", "seed"
]
missing_variables = [
    name for name in required_variables if name not in globals()
]
if missing_variables:
    raise NameError(
        f"Run the earlier notebook cells first. Missing: {missing_variables}"
    )

# Assignment settings
R = 30
B_FINAL = len(final_trees)  # Your existing final ensemble has 50 trees.

# Keep the clean test set fixed for all repetitions.
X_eval = X_test.copy()
y_eval = y_test.copy()

# (classes_q3, n_classes_q3, class_to_col_q3 and the q3_* helper
#  functions are defined in src/q3_helpers.py, unchanged.)

# Retrieve the SELECTED settings from your existing fitted models.
selected_tree_depth = best_tree.get_params()["max_depth"]
selected_tree_leaf = best_tree.get_params()["min_samples_leaf"]

selected_boost_depth = best_boost_model.max_depth
selected_boost_rounds = best_boost_model.n_estimators

print("Selected single-tree depth:", selected_tree_depth)
print("Selected single-tree min_samples_leaf:", selected_tree_leaf)
print("Selected boosting depth:", selected_boost_depth)
print("Selected boosting rounds:", selected_boost_rounds)
print("Final bagging ensemble size:", B_FINAL)


# Store all prediction vectors and test accuracies.
q3_vectors = {
    "Single tree": [],
    "Bagging": [],
    "Boosting": []
}

q3_accuracies = {
    "Single tree": [],
    "Bagging": [],
    "Boosting": []
}

# Independent RNG for the Q3 experiment.
rng_q3 = np.random.default_rng(seed + 3000)

for rep in range(R):

    # Outer bootstrap: six sampled faces per identity.
    outer_idx = q3_bootstrap_indices(y_train, rng_q3)
    X_boot = X_train[outer_idx]
    y_boot = y_train[outer_idx]

    # --------------------------------------------------------
    # 1. Selected single tree: keep its selected settings fixed
    # --------------------------------------------------------
    tree = DecisionTreeClassifier(
        max_depth=selected_tree_depth,
        min_samples_leaf=selected_tree_leaf,
        random_state=seed + 10000 + rep
    )
    tree.fit(X_boot, y_boot)

    p_tree = q3_single_tree_vector(tree, X_eval)
    q3_vectors["Single tree"].append(p_tree)

    q3_accuracies["Single tree"].append(
        accuracy_score(
            y_eval, classes_q3[np.argmax(p_tree, axis=1)]
        )
    )

    # --------------------------------------------------------
    # 2. Final bagged model: preserve its 50-tree ensemble size
    # --------------------------------------------------------
    refit_trees = []

    for b in range(B_FINAL):
        inner_rng = np.random.default_rng(
            seed + 20000 + rep * B_FINAL + b
        )

        # Each constituent tree bootstraps the current outer sample.
        inner_idx = inner_rng.choice(
            len(X_boot), size=len(X_boot), replace=True
        )

        bag_tree = DecisionTreeClassifier(
            max_depth=selected_tree_depth,
            min_samples_leaf=selected_tree_leaf,
            random_state=seed + 30000 + rep * B_FINAL + b
        )

        bag_tree.fit(X_boot[inner_idx], y_boot[inner_idx])
        refit_trees.append(bag_tree)

    p_bag = q3_bagging_vector(refit_trees, X_eval)
    q3_vectors["Bagging"].append(p_bag)

    q3_accuracies["Bagging"].append(
        accuracy_score(
            y_eval, classes_q3[np.argmax(p_bag, axis=1)]
        )
    )

    # --------------------------------------------------------
    # 3. Selected SAMME boosting model: freeze selected settings
    # --------------------------------------------------------
    boost = SAMMEBoosting(
        max_depth=selected_boost_depth,
        n_estimators=selected_boost_rounds,
        random_state=seed + 40000 + rep
    )
    boost.fit(X_boot, y_boot)

    p_boost = q3_boosting_vector(boost, X_eval)
    q3_vectors["Boosting"].append(p_boost)

    q3_accuracies["Boosting"].append(
        accuracy_score(
            y_eval, classes_q3[np.argmax(p_boost, axis=1)]
        )
    )

    if (rep + 1) % 5 == 0:
        print(f"Completed {rep + 1}/{R} repetitions")


# Calculate the measured quantities.
summary_a_rows = []

for model_name in q3_vectors:
    P = np.asarray(q3_vectors[model_name])
    acc = np.asarray(q3_accuracies[model_name])

    bias2, variance = q3_bias_variance(P, y_eval)

    summary_a_rows.append({
        "Model": model_name,
        "Bias squared": bias2,
        "Variance": variance,
        "Mean test accuracy": np.mean(acc),
        "SD test accuracy": np.std(acc, ddof=1)
    })

summary_a = pd.DataFrame(summary_a_rows)

print("\nQ3(a): Measured bias, variance and accuracy stability")
display(summary_a.round(5))
# --------------------------------------------------------------------

save_table(summary_a, "q3a_bias_variance")
save_table(pd.DataFrame(q3_accuracies).rename_axis("repetition").reset_index(), "q3a_accuracy_per_repetition")
finish()
