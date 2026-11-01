"""Q2(c): training diagnostics of the selected SAMME model and the hardest training face."""
from common import setup, finish, load_data_and_split, load_artifact, save_table

display = setup("q2c_samme_diagnostics")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import samme  # noqa: F401  (needed to unpickle SAMMEBoosting)

X, y, X_train, y_train, X_val, y_val, X_test, y_test, seed = load_data_and_split()

best_boost_model = load_artifact("q2b_best_boost_model", produced_by="q2b_samme_grid")["best_boost_model"]

# ---------------- original notebook code (unchanged) ----------------
selected_model = best_boost_model

round_numbers = np.arange(1, len(selected_model.estimators_) + 1)

weighted_errors = selected_model.estimator_errors_
learner_weights = selected_model.estimator_weights_

# Maximum sample weight before each corresponding learner was fitted.
max_sample_weights = np.array([
    np.max(weights)
    for weights in selected_model.weight_history_
])

# Total weight of the 10 heaviest samples before each round.
top10_weights = np.array([
    np.sort(weights)[-10:].sum()
    for weights in selected_model.weight_history_
])

# Plot 1: Weighted training error.
plt.figure(figsize=(7, 4))
plt.plot(round_numbers, weighted_errors, marker="o")
plt.xlabel("Boosting round")
plt.ylabel("Weighted training error")
plt.title("SAMME: Weighted Training Error")
plt.grid(True)
plt.show()

# Plot 2: Learner weight.
plt.figure(figsize=(7, 4))
plt.plot(round_numbers, learner_weights, marker="o")
plt.xlabel("Boosting round")
plt.ylabel("Learner weight (alpha)")
plt.title("SAMME: Learner Weight")
plt.grid(True)
plt.show()

# Plot 3: Maximum sample weight.
plt.figure(figsize=(7, 4))
plt.plot(round_numbers, max_sample_weights, marker="o")
plt.xlabel("Boosting round")
plt.ylabel("Maximum sample weight")
plt.title("SAMME: Maximum Training Sample Weight")
plt.grid(True)
plt.show()

# Plot 4: Total weight of the ten heaviest images.
plt.figure(figsize=(7, 4))
plt.plot(round_numbers, top10_weights, marker="o")
plt.xlabel("Boosting round")
plt.ylabel("Total weight of 10 heaviest images")
plt.title("SAMME: Weight Concentration on Hard Images")
plt.grid(True)
plt.show()

selected_model = best_boost_model

# Weight distribution after the final recorded round.
if len(selected_model.weight_history_) > 0:

    # Reconstruct the final normalised sample weights by replaying
    # the SAMME updates using the stored round predictions.
    final_sample_weights = np.ones(len(y_train)) / len(y_train)

    for alpha, predictions in zip(
        selected_model.estimator_weights_,
        selected_model.prediction_history_[:len(selected_model.estimators_)]
    ):

        incorrect = predictions != y_train

        final_sample_weights *= np.exp(alpha * incorrect)
        final_sample_weights /= final_sample_weights.sum()

    hard_index = int(np.argmax(final_sample_weights))

    print("Training sample index:", hard_index)
    print("True class:", y_train[hard_index])
    print("Final sample weight:", final_sample_weights[hard_index])
    print("Uniform initial weight:", 1 / len(y_train))

    # Prediction history across boosting rounds.
    history = []

    for t, tree in enumerate(selected_model.estimators_):

        pred = tree.predict(X_train[hard_index:hard_index + 1])[0]

        history.append({
            "round": t + 1,
            "true_class": y_train[hard_index],
            "predicted_class": pred,
            "correct": pred == y_train[hard_index],
            "learner_weight": selected_model.estimator_weights_[t]
        })

    history_df = pd.DataFrame(history)
    display(history_df)

    # Display the selected face.
    plt.figure(figsize=(4, 4))
    plt.imshow(X_train[hard_index].reshape(64, 64), cmap="gray")
    plt.title(
        f"High-weight training image\n"
        f"True class: {y_train[hard_index]}\n"
        f"Final weight: {final_sample_weights[hard_index]:.4f}"
    )
    plt.axis("off")
    plt.show()

    # Plot its weight after each completed boosting round.
    weight_after_round = []
    weights = np.ones(len(y_train)) / len(y_train)

    for alpha, predictions in zip(
        selected_model.estimator_weights_,
        selected_model.prediction_history_[:len(selected_model.estimators_)]
    ):

        incorrect = predictions != y_train
        weights *= np.exp(alpha * incorrect)
        weights /= weights.sum()

        weight_after_round.append(weights[hard_index])

    plt.figure(figsize=(7, 4))
    plt.plot(
        np.arange(1, len(weight_after_round) + 1),
        weight_after_round,
        marker="o"
    )
    plt.xlabel("Boosting round")
    plt.ylabel("Sample weight")
    plt.title("Weight History of the Selected Training Face")
    plt.grid(True)
    plt.show()
# --------------------------------------------------------------------

save_table(pd.DataFrame({
    "round": round_numbers,
    "weighted_error": weighted_errors,
    "learner_weight_alpha": learner_weights,
    "max_sample_weight": max_sample_weights,
    "top10_weight": top10_weights,
}), "q2c_samme_round_diagnostics")
if len(selected_model.weight_history_) > 0:
    save_table(history_df, "q2c_hard_sample_history")
finish()
