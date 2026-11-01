"""Q1(b): bagging with majority vote, accuracy vs number of trees B."""
from common import setup, finish, load_data_and_split, save_table

display = setup("q1b_bagging")

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y, X_train, y_train, X_val, y_val, X_test, y_test, seed = load_data_and_split()

# ---------------- original notebook code (unchanged) ----------------
t = 10

trees = []

for b in range(t):
    rng = np.random.default_rng(seed + b)

    sample_indices = rng.choice(len(X_train), size=len(X_train), replace=True)

    X_boost = X_train[sample_indices]
    y_boost = y_train[sample_indices]

    T = DecisionTreeClassifier(
        max_depth=32,
        min_samples_leaf= 2,
        random_state=seed + b
    )

    T.fit(X_boost, y_boost)

    trees.append(T)
print("Number of trees:", len(trees))


def bagging_predict(trees, X):
    all_predictions = []

    # Get predictions from every tree
    for tree in trees:
        predictions = tree.predict(X)
        all_predictions.append(predictions)

    # Convert to a NumPy array
    all_predictions = np.array(all_predictions)

    # Find the most common class for each example
    final_predictions = []

    for i in range(X.shape[0]):
        votes = all_predictions[:, i]
        classes, counts = np.unique(votes, return_counts=True)
        winner = classes[np.argmax(counts)]
        final_predictions.append(winner)

    return np.array(final_predictions)


val_predictions = bagging_predict(trees, X_val)

val_accuracy = accuracy_score(y_val, val_predictions)

print("Validation accuracy:", val_accuracy)

B_values = [1, 5, 10, 25, 50, 100, 200]

results = []

for t in B_values:

    trees = []
    distinct_counts = []

    for b in range(t):

        rng = np.random.default_rng(seed + b)

        sample_indices = rng.choice(
            len(X_train),
            size=len(X_train),
            replace=True
        )

        X_boot = X_train[sample_indices]
        y_boot = y_train[sample_indices]

        T = DecisionTreeClassifier(
            max_depth=None,       # Frozen setting from Q1(a)
            min_samples_leaf=2,   # Frozen setting from Q1(a)
            random_state=seed + b
        )

        T.fit(X_boot, y_boot)
        trees.append(T)

        distinct_counts.append(len(np.unique(sample_indices)))

    # Get predictions from every tree
    train_preds = np.array([tree.predict(X_train) for tree in trees])
    val_preds = np.array([tree.predict(X_val) for tree in trees])
    test_preds = np.array([tree.predict(X_test) for tree in trees])

    # Majority voting
    def majority_vote(preds):
        final_preds = []

        for i in range(preds.shape[1]):
            classes, counts = np.unique(preds[:, i], return_counts=True)
            final_preds.append(classes[np.argmax(counts)])

        return np.array(final_preds)

    train_final = majority_vote(train_preds)
    val_final = majority_vote(val_preds)
    test_final = majority_vote(test_preds)

    # Theoretical expected distinct examples
    n = len(X_train)
    theoretical = n * (1 - (1 - 1/n)**n)

    results.append({
        "B": t,
        "train_accuracy": accuracy_score(y_train, train_final),
        "validation_accuracy": accuracy_score(y_val, val_final),
        "test_accuracy": accuracy_score(y_test, test_final),
        "avg_distinct_examples": np.mean(distinct_counts),
        "theoretical_distinct_examples": theoretical
    })

results_df_bagging = pd.DataFrame(results)
display(results_df_bagging.round(4))
# --------------------------------------------------------------------

save_table(results_df_bagging, "q1b_bagging_vs_B")
finish()
