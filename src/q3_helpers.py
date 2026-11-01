"""Helper functions defined in the notebook's Q3(a) cell, unchanged.
Kept in a module because Q3(b) and Q3(c) reuse them."""

import numpy as np

from common import load_data_and_split

_, _, X_train, y_train, _, _, _, _, _ = load_data_and_split()

# ---------------- original notebook code (unchanged) ----------------
classes_q3 = np.unique(y_train)
n_classes_q3 = len(classes_q3)
class_to_col_q3 = {
    label: j for j, label in enumerate(classes_q3)
}


def q3_one_hot(y):
    """One-hot target vectors using the fixed class order."""
    output = np.zeros((len(y), n_classes_q3), dtype=float)

    for i, label in enumerate(y):
        output[i, class_to_col_q3[label]] = 1.0

    return output


def q3_bootstrap_indices(y, rng):
    """
    Draw six examples with replacement WITHIN EACH IDENTITY.
    The original training set must have six examples per identity.
    """
    selected = []

    for identity in classes_q3:
        identity_indices = np.flatnonzero(y == identity)

        if len(identity_indices) != 6:
            raise ValueError(
                f"Identity {identity} has {len(identity_indices)} "
                "training faces; this assignment expects six."
            )

        selected.extend(
            rng.choice(identity_indices, size=6, replace=True)
        )

    return np.asarray(selected, dtype=int)


def q3_align_proba(model, X):
    """Align predict_proba columns with classes_q3."""
    raw = model.predict_proba(X)
    aligned = np.zeros((len(X), n_classes_q3), dtype=float)

    for j, label in enumerate(model.classes_):
        aligned[:, class_to_col_q3[label]] = raw[:, j]

    return aligned


def q3_single_tree_vector(model, X):
    return q3_align_proba(model, X)


def q3_bagging_vector(trees, X):
    """Normalised class vote proportions from a bagged ensemble."""
    predictions = np.asarray([tree.predict(X) for tree in trees])
    votes = np.zeros((len(X), n_classes_q3), dtype=float)

    for j, label in enumerate(classes_q3):
        votes[:, j] = np.mean(predictions == label, axis=0)

    return votes


def q3_boosting_vector(model, X):
    """
    Use the normalised weighted vote returned by your
    existing SAMMEBoosting.predict_proba().
    """
    return q3_align_proba(model, X)


def q3_bias_variance(P, y_true):
    """
    P shape: (R, N, K)

    Bias^2 = mean_i ||mean_r p_r(x_i) - e_i||^2
    Variance = mean_i,r ||p_r(x_i) - mean_r p_r(x_i)||^2
    """
    p_mean = np.mean(P, axis=0)
    targets = q3_one_hot(y_true)

    bias_squared = np.mean(
        np.sum((p_mean - targets) ** 2, axis=1)
    )

    variance = np.mean(
        np.sum((P - p_mean[None, :, :]) ** 2, axis=2)
    )

    return float(bias_squared), float(variance)
