"""Q2(a): masked test set check and a hand-worked SAMME round on a toy problem."""
from common import setup, finish, load_data_and_split

display = setup("q2a_samme_demo")

X, y, X_train, y_train, X_val, y_val, X_test, y_test, seed = load_data_and_split()

# ---------------- original notebook code (unchanged) ----------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Ensure the masked test set exists.
# Each image is 64 x 64, stored as a flattened vector of 4096 pixels.

X_test_masked = X_test.reshape(-1, 64, 64).copy()
X_test_masked[:, 24:40, 24:40] = 0
X_test_masked = X_test_masked.reshape(-1, 4096)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Clean test samples:", len(X_test))
print("Masked test samples:", len(X_test_masked))
print("Number of classes:", len(np.unique(y_train)))

# Nine samples, three classes: 0, 1 and 2.
X_demo = np.array([
    [0.0], [1.0], [2.0], [3.0],
    [4.0], [5.0], [6.0], [7.0], [8.0]
])

y_demo = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2])

K_demo = len(np.unique(y_demo))
n_demo = len(y_demo)

# Initially, every sample has equal weight.
w_demo = np.ones(n_demo) / n_demo

print("Initial sample weights:")
print(w_demo)

# Train a weak learner using the initial sample weights.
demo_tree = DecisionTreeClassifier(
    max_depth=1,
    random_state=seed
)

demo_tree.fit(X_demo, y_demo, sample_weight=w_demo)

demo_pred = demo_tree.predict(X_demo)

print("\nTrue labels:")
print(y_demo)

print("\nWeak learner predictions:")
print(demo_pred)

# Weighted classification error.
incorrect = (demo_pred != y_demo)

error = np.sum(w_demo * incorrect)

print("\nWeighted error:", error)

# A multiclass random-guessing baseline has error 1 - 1/K.
error_threshold = 1 - 1 / K_demo

print("Random-guessing error threshold:", error_threshold)

if error >= error_threshold:
    print("\nWeak learner is not better than random.")
    print("SAMME stops; this learner is not added to the ensemble.")

else:
    # Avoid log(0) if a learner perfectly classifies every sample.
    error_for_alpha = max(error, 1e-12)

    alpha = (
        np.log((1 - error_for_alpha) / error_for_alpha)
        + np.log(K_demo - 1)
    )

    print("\nLearner weight alpha:", alpha)

    # Increase the weights of incorrectly classified samples.
    updated_weights = w_demo * np.exp(alpha * incorrect)

    # Normalise so that weights sum to 1.
    updated_weights = updated_weights / updated_weights.sum()

    print("\nUpdated normalised sample weights:")
    print(updated_weights)

    print("\nSum of updated weights:", updated_weights.sum())
# --------------------------------------------------------------------

finish()
