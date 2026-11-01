"""Q1(d): final 50-tree bagged model vs single tree on clean and masked test faces."""
from common import setup, finish, load_data_and_split, load_artifact, save_artifact, save_table

display = setup("q1d_masked_robustness")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y, X_train, y_train, X_val, y_val, X_test, y_test, seed = load_data_and_split()

_q1a = load_artifact("q1a_best_tree", produced_by="q1a_tree_grid")
best_tree = _q1a["best_tree"]
best_depth = _q1a["best_depth"]
best_leaf = _q1a["best_leaf"]

# ---------------- original notebook code (unchanged) ----------------
# Q1(d): Final model uses B = 50, selected using validation accuracy
B_final = 50

# Create masked test images: rows 24-39 and columns 24-39 set to zero
X_test_masked = X_test.reshape(-1, 64, 64).copy()
X_test_masked[:, 24:40, 24:40] = 0
X_test_masked = X_test_masked.reshape(-1, 4096)

# Train the final bagged model using the frozen settings
final_trees = []

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

    tree.fit(X_train[sample_indices], y_train[sample_indices])
    final_trees.append(tree)

print("Final number of trees:", len(final_trees))


def evaluate_models(X_eval, y_eval, dataset_name):
    # Single decision tree
    tree_pred = best_tree.predict(X_eval)
    tree_prob = best_tree.predict_proba(X_eval)
    tree_conf = np.max(tree_prob, axis=1)

    # Bagging
    all_votes = np.array([
        tree.predict(X_eval) for tree in final_trees
    ])

    bag_pred = []
    bag_conf = []

    for i in range(len(X_eval)):
        classes, counts = np.unique(
            all_votes[:, i], return_counts=True
        )

        winner_index = np.argmax(counts)

        bag_pred.append(classes[winner_index])
        bag_conf.append(counts[winner_index] / len(final_trees))

    bag_pred = np.array(bag_pred)
    bag_conf = np.array(bag_conf)

    return {
        "Dataset": dataset_name,
        "Tree accuracy": accuracy_score(y_eval, tree_pred),
        "Tree mean confidence": np.mean(tree_conf),
        "Bagging accuracy": accuracy_score(y_eval, bag_pred),
        "Bagging mean confidence": np.mean(bag_conf)
    }


results_d = [
    evaluate_models(X_test, y_test, "Clean test"),
    evaluate_models(X_test_masked, y_test, "Masked test")
]

display(pd.DataFrame(results_d).round(4))


def get_predictions_and_votes(X_eval):
    tree_predictions = best_tree.predict(X_eval)

    all_votes = np.array([
        tree.predict(X_eval) for tree in final_trees
    ])

    bag_predictions = []
    vote_counts = []

    for i in range(len(X_eval)):
        classes, counts = np.unique(
            all_votes[:, i], return_counts=True
        )

        bag_predictions.append(classes[np.argmax(counts)])

        vote_counts.append({
            int(cls): int(count)
            for cls, count in zip(classes, counts)
        })

    return np.array(tree_predictions), np.array(bag_predictions), vote_counts


# Search both clean and masked test sets
corrected_cases = []
both_wrong_cases = []

for dataset_name, X_eval in [
    ("Clean test", X_test),
    ("Masked test", X_test_masked)
]:
    tree_pred, bag_pred, vote_counts = get_predictions_and_votes(X_eval)

    for i in range(len(y_test)):
        true_label = int(y_test[i])
        tree_label = int(tree_pred[i])
        bag_label = int(bag_pred[i])

        case = {
            "dataset": dataset_name,
            "index": i,
            "true_label": true_label,
            "tree_prediction": tree_label,
            "bagging_prediction": bag_label,
            "bagging_vote_counts": vote_counts[i]
        }

        # Bagging corrects a mistake made by the single tree
        if tree_label != true_label and bag_label == true_label:
            corrected_cases.append(case)

        # Both models predict the wrong identity
        if tree_label != true_label and bag_label != true_label:
            both_wrong_cases.append(case)

        if len(corrected_cases) >= 2 and len(both_wrong_cases) >= 2:
            break

    if len(corrected_cases) >= 2 and len(both_wrong_cases) >= 2:
        break


print("=== Two cases where bagging corrects the tree ===")
for case in corrected_cases[:2]:
    print(case)

print("\n=== Two cases where both models are wrong ===")
for case in both_wrong_cases[:2]:
    print(case)

selected_cases = corrected_cases[:2] + both_wrong_cases[:2]

fig, axes = plt.subplots(1, len(selected_cases), figsize=(14, 4))

for ax, case in zip(axes, selected_cases):
    i = case["index"]

    if case["dataset"] == "Clean test":
        face = X_test[i].reshape(64, 64)
    else:
        face = X_test_masked[i].reshape(64, 64)

    ax.imshow(face, cmap="gray")
    ax.set_title(
        f'{case["dataset"]}\n'
        f'True: {case["true_label"]}\n'
        f'Tree: {case["tree_prediction"]}\n'
        f'Bagging: {case["bagging_prediction"]}'
    )
    ax.axis("off")

plt.tight_layout()
plt.show()
# --------------------------------------------------------------------

save_table(pd.DataFrame(results_d), "q1d_clean_vs_masked")
save_artifact("q1d_final_trees", {"final_trees": final_trees, "B_final": B_final})
finish()
