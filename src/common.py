"""
Shared helpers for every question script.

Nothing here changes the experiment logic from the notebook. It only:
  * tees everything printed to the console into logs/<part>.log
  * replaces notebook display() with a plain-text table print
  * saves every figure that the notebook showed with plt.show() into figures/
  * loads the saved train/val/test split from data/olivetti_split.npz
  * saves / loads fitted objects in models/ so later parts can reuse earlier ones
"""

import json
import os
import platform
import sys
import time
from datetime import datetime

import matplotlib

matplotlib.use("Agg")  # no window; figures are written to disk
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(ROOT, "configs", "config.json")) as fh:
    CONFIG = json.load(fh)

DATA_DIR = os.path.join(ROOT, CONFIG["paths"]["data"])
FIG_DIR = os.path.join(ROOT, CONFIG["paths"]["figures"])
LOG_DIR = os.path.join(ROOT, CONFIG["paths"]["logs"])
MODEL_DIR = os.path.join(ROOT, CONFIG["paths"]["models"])
RESULT_DIR = os.path.join(ROOT, CONFIG["paths"]["results"])
SPLIT_FILE = os.path.join(ROOT, CONFIG["split"]["file"])

for _d in (DATA_DIR, FIG_DIR, LOG_DIR, MODEL_DIR, RESULT_DIR):
    os.makedirs(_d, exist_ok=True)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 500)


class _Tee:
    """Write to the console and the log file at the same time."""

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, text):
        self.stream.write(text)
        self.log_file.write(text)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()


_PART = {"name": None, "fig": 0, "t0": None}


def _show_and_save(*args, **kwargs):
    """Replacement for plt.show(): save the current figure, then close it."""
    _PART["fig"] += 1
    stem = f"{_PART['name']}_fig{_PART['fig']:02d}"
    fig = plt.gcf()
    fig.savefig(os.path.join(FIG_DIR, stem + ".png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, stem + ".pdf"), bbox_inches="tight")
    print(f"[figure saved] figures/{stem}.png")
    plt.close(fig)


def display(obj):
    """Plain-text replacement for the notebook's display()."""
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        print(obj.to_string())
    else:
        print(obj)


def setup(part_name):
    """Start logging for one question part. Call once at the top of a script."""
    _PART["name"] = part_name
    _PART["fig"] = 0
    _PART["t0"] = time.perf_counter()

    log_path = os.path.join(LOG_DIR, f"{part_name}.log")
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    import sklearn
    import matplotlib as mpl

    print("=" * 70)
    print(f"Part        : {part_name}")
    print(f"Started     : {datetime.now().isoformat(timespec='seconds')}")
    print(f"Python      : {platform.python_version()} ({platform.platform()})")
    print(f"numpy       : {np.__version__}")
    print(f"pandas      : {pd.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"matplotlib  : {mpl.__version__}")
    print(f"Seed        : {CONFIG['seed']} (roll {CONFIG['roll_number']})")
    print("=" * 70)

    plt.show = _show_and_save
    return display


def finish():
    """Print the run time at the end of a script."""
    elapsed = time.perf_counter() - _PART["t0"]
    print("=" * 70)
    print(f"Finished {_PART['name']} in {elapsed:.1f} s")
    print("=" * 70)


def load_data_and_split():
    """
    Load Olivetti faces and the SAVED split (data/olivetti_split.npz,
    written by src/q0_data_split.py). Returns the same variables the
    notebook used.
    """
    from sklearn.datasets import fetch_olivetti_faces

    if not os.path.exists(SPLIT_FILE):
        raise FileNotFoundError(
            f"{SPLIT_FILE} not found. Run `python src/q0_data_split.py` first."
        )

    f = fetch_olivetti_faces()
    X = f.data
    y = f.target

    split = np.load(SPLIT_FILE)
    train_indices = split["train_idx"]
    validation_indices = split["val_idx"]
    test_indices = split["test_idx"]

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_val = X[validation_indices]
    y_val = y[validation_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    seed = int(CONFIG["roll"])

    return X, y, X_train, y_train, X_val, y_val, X_test, y_test, seed


def save_artifact(name, obj):
    path = os.path.join(MODEL_DIR, f"{name}.joblib")
    joblib.dump(obj, path)
    print(f"[artifact saved] models/{name}.joblib")


def load_artifact(name, produced_by):
    path = os.path.join(MODEL_DIR, f"{name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"models/{name}.joblib not found. Run `python src/{produced_by}.py` first."
        )
    print(f"[artifact loaded] models/{name}.joblib")
    return joblib.load(path)


def save_table(df, name):
    path = os.path.join(RESULT_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"[table saved] results/{name}.csv")
