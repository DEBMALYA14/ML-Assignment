"""Q2(b): SAMME boosting, grid over weak-learner depth x number of rounds."""
from common import setup, finish, load_data_and_split, save_artifact, save_table

display = setup("q2b_samme_grid")

import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from samme import SAMMEBoosting  # class body is unchanged from the notebook

X, y, X_train, y_train, X_val, y_val, X_test, y_test, seed = load_data_and_split()

# ---------------- original notebook code (unchanged) ----------------
depth_values = [1, 2, 3, 4]
round_values = [10, 25, 50, 100]

boosting_grid_results = []
boosting_models = {}

for depth in depth_values:

    for rounds in round_values:

        print(f"Training depth={depth}, rounds={rounds}")

        model = SAMMEBoosting(
            max_depth=depth,
            n_estimators=rounds,
            random_state=seed
        )

        start_time = time.perf_counter()

        model.fit(X_train, y_train)

        training_time = time.perf_counter() - start_time

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        train_accuracy = accuracy_score(y_train, train_pred)
        val_accuracy = accuracy_score(y_val, val_pred)

        boosting_models[(depth, rounds)] = model

        boosting_grid_results.append({
            "depth": depth,
            "requested_rounds": rounds,
            "actual_rounds": len(model.estimators_),
            "train_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
            "training_time_seconds": training_time
        })

boosting_grid_df = pd.DataFrame(boosting_grid_results)

boosting_grid_df = boosting_grid_df.sort_values(
    ["validation_accuracy", "depth", "requested_rounds"],
    ascending=[False, True, True]
).reset_index(drop=True)

display(boosting_grid_df)

# Choose using validation accuracy only.
best_boost_row = boosting_grid_df.iloc[0]

best_boost_depth = int(best_boost_row["depth"])
best_boost_rounds = int(best_boost_row["requested_rounds"])

best_boost_model = boosting_models[
    (best_boost_depth, best_boost_rounds)
]

print("Selected depth:", best_boost_depth)
print("Selected requested rounds:", best_boost_rounds)
print("Actual learners:", len(best_boost_model.estimators_))
print("Validation accuracy:", best_boost_row["validation_accuracy"])
# --------------------------------------------------------------------

save_table(boosting_grid_df, "q2b_samme_grid")
save_artifact("q2b_best_boost_model", {
    "best_boost_model": best_boost_model,
    "best_boost_depth": best_boost_depth,
    "best_boost_rounds": best_boost_rounds,
})
finish()
