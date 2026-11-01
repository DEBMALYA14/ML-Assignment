"""Q1(c): OOB accuracy, pairwise disagreement and vote margin vs B; choose B."""
from common import setup, finish, load_data_and_split, load_artifact, save_table

display = setup("q1c_oob_diversity")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y, X_train, y_train, X_val, y_val, X_test, y_test, seed = load_data_and_split()

_q1a = load_artifact("q1a_best_tree", produced_by="q1a_tree_grid")
best_depth = _q1a["best_depth"]
best_leaf = _q1a["best_leaf"]

# ---------------- original notebook code (unchanged) ----------------
B_values = [1, 5, 10, 25, 50, 100, 200]
results_c = []

for B in B_values:
    trees = []
    oob_masks = []

    n = len(X_train)

    # Train B trees
    for b in range(B):
        rng = np.random.default_rng(seed + b)

        sample_indices = rng.choice(
            n, size=n, replace=True
        )

        # Remember which examples were NOT sampled
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[sample_indices] = False
        oob_masks.append(oob_mask)

        tree = DecisionTreeClassifier(
            max_depth=best_depth,
            min_samples_leaf=best_leaf,
            random_state=seed + b
        )

        tree.fit(X_train[sample_indices], y_train[sample_indices])
        trees.append(tree)

    # Predictions on training and validation data
    train_preds = np.array([tree.predict(X_train) for tree in trees])
    val_preds = np.array([tree.predict(X_val) for tree in trees])
    oob_masks = np.array(oob_masks)

    # 1. OOB accuracy
    oob_true = []
    oob_pred = []

    for i in range(n):
        eligible = oob_masks[:, i]

        if np.any(eligible):
            votes = train_preds[eligible, i]
            classes, counts = np.unique(votes, return_counts=True)
            oob_pred.append(classes[np.argmax(counts)])
            oob_true.append(y_train[i])

    oob_accuracy = accuracy_score(oob_true, oob_pred)

    # 2. Average pairwise disagreement
    disagreements = []

    for i in range(B):
        for j in range(i + 1, B):
            disagreements.append(
                np.mean(train_preds[i] != train_preds[j])
            )

    avg_disagreement = (
        np.mean(disagreements) if disagreements else np.nan
    )

    # 3. Mean vote margin: winning fraction - runner-up fraction
    margins = []

    for i in range(n):
        votes = train_preds[:, i]
        _, counts = np.unique(votes, return_counts=True)
        counts = np.sort(counts)[::-1]

        winning_votes = counts[0]
        runner_up_votes = counts[1] if len(counts) > 1 else 0

        margins.append((winning_votes - runner_up_votes) / B)

    mean_vote_margin = np.mean(margins)

    # Validation accuracy for choosing B
    val_final = []

    for i in range(len(X_val)):
        classes, counts = np.unique(val_preds[:, i], return_counts=True)
        val_final.append(classes[np.argmax(counts)])

    val_accuracy = accuracy_score(y_val, val_final)

    results_c.append({
        "B": B,
        "OOB_accuracy": oob_accuracy,
        "pairwise_disagreement": avg_disagreement,
        "mean_vote_margin": mean_vote_margin,
        "validation_accuracy": val_accuracy
    })

results_df_c = pd.DataFrame(results_c)
display(results_df_c.round(4))

plt.figure(figsize=(7, 4))
plt.plot(results_df_c["B"], results_df_c["OOB_accuracy"], marker="o")
plt.xlabel("Number of trees (B)")
plt.ylabel("OOB accuracy")
plt.title("OOB Accuracy vs B")
plt.grid(True)
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(
    results_df_c["B"],
    results_df_c["pairwise_disagreement"],
    marker="o"
)
plt.xlabel("Number of trees (B)")
plt.ylabel("Average pairwise disagreement")
plt.title("Pairwise Disagreement vs B")
plt.grid(True)
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(
    results_df_c["B"],
    results_df_c["mean_vote_margin"],
    marker="o"
)
plt.xlabel("Number of trees (B)")
plt.ylabel("Mean vote margin")
plt.title("Mean Vote Margin vs B")
plt.grid(True)
plt.show()

best_result = results_df_c.loc[
    results_df_c["validation_accuracy"].idxmax()
]

print("Selected B:", int(best_result["B"]))
print("Validation accuracy:", best_result["validation_accuracy"])
# --------------------------------------------------------------------

save_table(results_df_c, "q1c_oob_diversity")
finish()
