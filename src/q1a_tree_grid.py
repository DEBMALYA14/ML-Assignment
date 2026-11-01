"""Q1(a): single decision tree, grid search over max_depth x min_samples_leaf."""
from common import setup, finish, load_data_and_split, save_artifact, save_table

display = setup("q1a_tree_grid")

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y, X_train, y_train, X_val, y_val, X_test, y_test, seed = load_data_and_split()

# ---------------- original notebook code (unchanged) ----------------
depth = [2,4,8,16,None]
l_size = [1,2,4]

results = []

for depth in depth:
    for leaf_size in l_size:

        model = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_leaf=leaf_size,
            random_state=seed
        )

        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)

        results.append({
            "max_depth": depth,
            "min_samples_leaf": leaf_size,
            "train_accuracy": train_acc,
            "validation_accuracy": val_acc
        })


results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    "validation_accuracy", ascending=False
)

display(results_df)

best_row = results_df.iloc[0]

best_depth = best_row["max_depth"]
best_leaf = int(best_row["min_samples_leaf"])

# Convert NaN to None
if pd.isna(best_depth):
    best_depth = None

best_tree = DecisionTreeClassifier(
    max_depth=best_depth,
    min_samples_leaf=best_leaf,
    random_state=seed
)


best_tree.fit(X_train, y_train)

print("Original tree depth:", best_tree.get_depth())
print("Best depth:", best_depth)
print("Best leaf size:", best_leaf)
print("Best validation accuracy:", best_row["validation_accuracy"])
# --------------------------------------------------------------------

save_table(results_df, "q1a_tree_grid")
save_artifact("q1a_best_tree", {
    "best_tree": best_tree,
    "best_depth": best_depth,
    "best_leaf": best_leaf,
    "best_validation_accuracy": float(best_row["validation_accuracy"]),
})
finish()
