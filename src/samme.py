"""SAMMEBoosting class from Q2(b), unchanged. Kept in its own module so that
Q2(b), Q2(c), Q2(d) and Q3(a) can all import it and load the saved model."""

import numpy as np
from sklearn.tree import DecisionTreeClassifier


class SAMMEBoosting:

    def __init__(self, max_depth=1, n_estimators=50, random_state=0):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []

        self.weight_history_ = []
        self.prediction_history_ = []

        self.classes_ = None
        self.n_classes_ = None

        self.stopped_reason_ = None

    def fit(self, X, y):

        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        n_samples = len(y)

        # Equal initial weights.
        sample_weights = np.ones(n_samples) / n_samples

        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []
        self.weight_history_ = []
        self.prediction_history_ = []
        self.stopped_reason_ = None

        random_baseline_error = 1 - 1 / self.n_classes_

        for t in range(self.n_estimators):

            # Record weights before fitting the current learner.
            self.weight_history_.append(sample_weights.copy())

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                random_state=self.random_state + t
            )

            # Weighted fitting is essential for AdaBoost.
            tree.fit(
                X,
                y,
                sample_weight=sample_weights
            )

            predictions = tree.predict(X)

            # Weighted error of this weak learner.
            incorrect = (predictions != y)

            error = np.sum(sample_weights * incorrect)

            # Store predictions for later error analysis.
            self.prediction_history_.append(predictions.copy())

            # Stop if the weak learner is no better than random.
            if error >= random_baseline_error:
                self.stopped_reason_ = (
                    f"Round {t + 1}: weighted error {error:.6f} "
                    f"is not below the random baseline "
                    f"{random_baseline_error:.6f}."
                )

                # This learner is deliberately not added.
                self.weight_history_.pop()
                break

            # Prevent log(0) for a perfect learner.
            error_for_alpha = max(error, 1e-12)

            alpha = (
                np.log((1 - error_for_alpha) / error_for_alpha)
                + np.log(self.n_classes_ - 1)
            )

            # Keep the learner.
            self.estimators_.append(tree)
            self.estimator_weights_.append(alpha)
            self.estimator_errors_.append(error)

            # SAMME update:
            # incorrect examples are multiplied by exp(alpha).
            sample_weights *= np.exp(alpha * incorrect)

            # Normalise weights to sum to one.
            sample_weights /= sample_weights.sum()

            # A perfect learner ends the training process.
            if error <= 1e-12:
                self.stopped_reason_ = (
                    f"Perfect learner reached at round {t + 1}."
                )
                break

        self.estimator_weights_ = np.asarray(
            self.estimator_weights_, dtype=float
        )

        self.estimator_errors_ = np.asarray(
            self.estimator_errors_, dtype=float
        )

        if len(self.estimators_) == 0:
            raise ValueError(
                "SAMME could not find a weak learner better than random. "
                "Try another depth or inspect the data."
            )

        if self.stopped_reason_ is not None:
            print(self.stopped_reason_)

        return self

    def decision_scores(self, X):

        X = np.asarray(X)

        scores = np.zeros(
            (len(X), self.n_classes_),
            dtype=float
        )

        for tree, alpha in zip(
            self.estimators_,
            self.estimator_weights_
        ):

            predictions = tree.predict(X)

            for class_index, class_label in enumerate(self.classes_):
                scores[:, class_index] += (
                    alpha * (predictions == class_label)
                )

        return scores

    def predict_proba(self, X):

        scores = self.decision_scores(X)

        total_weight = self.estimator_weights_.sum()

        if total_weight <= 0:
            raise ValueError("Total learner weight must be positive.")

        # Assignment-required normalised weighted vote.
        return scores / total_weight

    def predict(self, X):

        scores = self.decision_scores(X)

        return self.classes_[np.argmax(scores, axis=1)]
