"""Data loading, sample faces, and the fixed 6/2/2 per-identity split.

Writes data/olivetti_split.npz, which every other script loads.
"""
import os

from common import setup, finish, save_table, SPLIT_FILE

display = setup("q0_data_split")

# ---------------- original notebook code (unchanged) ----------------
import re
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

f = fetch_olivetti_faces()
X = f.data
y = f.target

print(X.shape)
print(y.shape)
print("Number of classes:", len(np.unique(y)))

fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i, ax in enumerate(axes.flat):
    ax.imshow(X[i+2].reshape(64, 64), cmap="gray")
    ax.set_title(f"Identity: {y[i]}")
    ax.axis("off")

plt.tight_layout()
plt.show()

#My roll number is P26CS0002
roll = '260002'
seed = int(roll)
r = np.random.default_rng(seed)


train_indices = []
test_indices = []
validation_indices = []

for i in range(40):
    i_pos = np.where(y == i)[0].copy()
    r.shuffle(i_pos)

    for p in i_pos[:6]:
        train_indices.append(p)

    for p in i_pos[6:8]:
        test_indices.append(p)

    for p in i_pos[8:10]:
        validation_indices.append(p)


train_indices = np.array(train_indices)
test_indices = np.array(test_indices)
validation_indices = np.array(validation_indices)

X_train = X[train_indices]
y_train = y[train_indices]

X_val = X[validation_indices]
y_val = y[validation_indices]

X_test = X[test_indices]
y_test = y[test_indices]


np.savez(
    SPLIT_FILE,  # notebook: "olivetti_split.npz" (now kept in data/)
    train_idx = train_indices,
    val_idx = validation_indices,
    test_idx = test_indices
)

print("Training images:", len(X_train))
print("Validation images:", len(X_val))
print("Testing images:", len(X_test))
# --------------------------------------------------------------------

print("Split saved to data/olivetti_split.npz")
split_table = pd.DataFrame({
    "split": ["train"] * len(train_indices) + ["validation"] * len(validation_indices) + ["test"] * len(test_indices),
    "dataset_index": np.concatenate([train_indices, validation_indices, test_indices]),
    "identity": np.concatenate([y_train, y_val, y_test]),
})
save_table(split_table, "q0_split_indices")
finish()
