"""Run every question script in order (works on Windows, macOS and Linux).

    python run_all.py              # everything
    python run_all.py q2b q2c      # only the parts whose names start with these
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PARTS = [
    "q0_data_split",
    "q1a_tree_grid",
    "q1b_bagging",
    "q1c_oob_diversity",
    "q1d_masked_robustness",
    "q2a_samme_demo",
    "q2b_samme_grid",
    "q2c_samme_diagnostics",
    "q2d_compare_models",
    "q3a_bias_variance",
    "q3b_depth_bias_variance",
    "q3c_depth_diagnostics",
]

selected = [p for p in PARTS if not sys.argv[1:] or any(p.startswith(a) for a in sys.argv[1:])]

t0 = time.perf_counter()
for part in selected:
    print(f"\n>>> python src/{part}.py", flush=True)
    result = subprocess.run([sys.executable, str(ROOT / "src" / f"{part}.py")], cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"{part} failed (exit code {result.returncode}); see logs/{part}.log")
print(f"\nAll done in {time.perf_counter() - t0:.0f} s. Logs: logs/  Figures: figures/  Tables: results/")
