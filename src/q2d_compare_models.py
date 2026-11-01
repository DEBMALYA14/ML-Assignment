"""Q2(d): single tree vs bagging vs SAMME boosting on clean and masked test faces."""
from common import setup, finish, load_data_and_split, load_artifact, save_table

display = setup("q2d_compare_models")

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import samme  # noqa: F401  (needed to unpickle SAMMEBoosting)

X, y, X_train, y_train, X_val, y_val, X_test, y_test, seed = load_data_and_split()

_q1a = load_artifact("q1a_best_tree", produced_by="q1a_tree_grid")
best_tree = _q1a["best_tree"]
best_depth = _q1a["best_depth"]
best_leaf = _q1a["best_leaf"]
final_trees = load_artifact("q1d_final_trees", produced_by="q1d_masked_robustness")["final_trees"]
best_boost_model = load_artifact("q2b_best_boost_model", produced_by="q2b_samme_grid")["best_boost_model"]

# Masked test set, created exactly as in Q1(d) / Q2(a).
X_test_masked = X_test.reshape(-1, 64, 64).copy()
X_test_masked[:, 24:40, 24:40] = 0
X_test_masked = X_test_masked.reshape(-1, 4096)

# ---------------- original notebook code (unchanged) ----------------
B_final = 50

# Reuse the final Q1 bagged model if it is available and has 50 trees.
if "final_trees" in globals() and len(final_trees) == B_final:

    final_bagged_trees = final_trees

else:

    final_bagged_trees = []

    for b in range(B_final):

        rng = np.random.default_rng(seed + b)

        sample_indices = rng.choice(
            len(X_train),
            size=len(X_train),
            replace=True
        )

        tree = DecisionTreeClassifier(
            max_depth=best_depth,
            min_samples_leaf=best_leaf,
            random_state=seed + b
        )

        tree.fit(
            X_train[sample_indices],
            y_train[sample_indices]
        )

        final_bagged_trees.append(tree)

print("Single tree depth:", best_tree.get_depth())
print("Number of bagged trees:", len(final_bagged_trees))
print("Number of boosting learners:", len(best_boost_model.estimators_))


def get_bagging_votes(trees, X):

    # Shape: (number of trees, number of samples)
    predictions = np.array([
        tree.predict(X) for tree in trees
    ])

    classes = np.unique(y_train)

    votes = np.zeros(
        (len(X), len(classes)),
        dtype=float
    )

    for j, class_label in enumerate(classes):
        votes[:, j] = np.sum(
            predictions == class_label,
            axis=0
        )

    return classes, votes


def evaluate_q2_models(X_eval, y_eval, dataset_name):

    # ----------------------------
    # Single tree
    # ----------------------------
    tree_pred = best_tree.predict(X_eval)

    tree_prob = best_tree.predict_proba(X_eval)

    tree_confidence = np.max(tree_prob, axis=1)

    # ----------------------------
    # Bagging
    # ----------------------------
    bag_classes, bag_votes = get_bagging_votes(
        final_bagged_trees, X_eval
    )

    bag_pred = bag_classes[np.argmax(bag_votes, axis=1)]

    bag_confidence = np.max(bag_votes, axis=1) / len(final_bagged_trees)

    # ----------------------------
    # Boosting
    # ----------------------------
    boost_scores = best_boost_model.decision_scores(X_eval)

    boost_classes = best_boost_model.classes_

    boost_pred = boost_classes[np.argmax(boost_scores, axis=1)]

    boost_q = best_boost_model.predict_proba(X_eval)

    boost_confidence = np.max(boost_q, axis=1)

    rows = [
        {
            "dataset": dataset_name,
            "model": "Single tree",
            "accuracy": accuracy_score(y_eval, tree_pred),
            "mean_winning_confidence": np.mean(tree_confidence)
        },
        {
            "dataset": dataset_name,
            "model": "Bagging",
            "accuracy": accuracy_score(y_eval, bag_pred),
            "mean_winning_confidence": np.mean(bag_confidence)
        },
        {
            "dataset": dataset_name,
            "model": "Boosting",
            "accuracy": accuracy_score(y_eval, boost_pred),
            "mean_winning_confidence": np.mean(boost_confidence)
        }
    ]

    return pd.DataFrame(rows)


clean_results = evaluate_q2_models(
    X_test, y_test, "Clean test"
)

masked_results = evaluate_q2_models(
    X_test_masked, y_test, "Masked test"
)

q2d_results = pd.concat(
    [clean_results, masked_results],
    ignore_index=True
)

display(q2d_results)
# --------------------------------------------------------------------

save_table(q2d_results, "q2d_model_comparison")
finish()
