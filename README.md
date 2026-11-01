# ML Assignment 2: Decision Trees, Bagging and SAMME Boosting

**Debmalya Bhattacharya** · Roll No. **P26CS0002** · IIT Jodhpur

Face identification on the **Olivetti Faces** dataset (40 people, 400 grey-scale 64×64 images) using a single decision tree, a bagged ensemble, and a multiclass AdaBoost (SAMME) implemented from scratch. A bias–variance analysis explains the results.

---

## Repository contents

| File | Description |
|---|---|
| `a.ipynb` | Full implementation and all experiments (Q1–Q3) |
| `ML_Assignment.pdf` | Report with results, plots and explanations |
| `olivetti_split.npz` | Saved train/val/test indices, so the split is reproducible |

## Setup

```bash
pip install numpy pandas scikit-learn matplotlib
jupyter notebook a.ipynb    # then Run All
```

The dataset downloads automatically through `sklearn.datasets.fetch_olivetti_faces()`.

## Data split

For each identity, the 10 images are shuffled using the roll number as seed (`seed = 260002`) and split **6 / 2 / 2**:

| Split | Images |
|---|---|
| Train | 240 |
| Validation | 80 |
| Test | 80 |

All hyperparameters are chosen on the validation set. The test set is used only for the final comparison.

## What is implemented

- **Q1: Single tree and bagging**
  - Grid search over `max_depth` and `min_samples_leaf`
  - Bagging with B ∈ {1, 5, 10, 25, 50, 100, 200}
  - Out-of-bag (OOB) accuracy, pairwise tree disagreement and vote margins
  - Robustness test with the central 16×16 region of each test face masked
- **Q2: SAMME boosting (from scratch)**
  - One round worked by hand
  - Grid search over depth and number of rounds
  - Learner-weight and sample-weight dynamics, and the hardest training face
- **Q3: Bias–variance analysis**
  - 30 stratified bootstrap repetitions
  - Measured bias² and variance for each model
  - Effect of tree depth, leaf occupancy and tree disagreement

## Results

**Selected models**

| Model | Settings | Validation acc. |
|---|---|---|
| Single tree | `max_depth=None`, `min_samples_leaf=2` (grows to depth 32) | 0.6125 |
| Bagging | B = 50 | 0.7875 |
| SAMME | depth 4, 100 rounds | 0.5250 |

**Test accuracy (clean vs. masked faces)**

| Model | Clean | Masked |
|---|---|---|
| Single tree | 0.6250 | 0.0750 |
| **Bagging (B=50)** | **0.8125** | **0.7625** |
| SAMME boosting | 0.4625 | 0.4500 |

**Bias–variance (30 repetitions)**

| Model | Bias² | Variance | Mean test acc. |
|---|---|---|---|
| Single tree | 0.374 | 0.642 | 0.492 |
| Bagging | 0.501 | 0.031 | 0.785 |
| SAMME | 0.892 | 0.016 | 0.268 |

## Key findings

- A fully grown tree has **low bias and very high variance**. It is also badly overconfident: on masked faces it reports 100% confidence while being right only 7.5% of the time.
- **Bagging removes about 96% of the tree's variance**. That makes it the most accurate model and by far the most robust to occlusion.
- Bagging helps **deep trees a lot (+35 points) but shallow trees barely (+7)**. Bagging reduces variance, not bias.
- **SAMME** fits the training set perfectly but generalises poorly here. With only 6 images per person in 4096 dimensions, the main source of error is variance, not bias.

> Bag low-bias, high-variance learners; boost high-bias, low-variance ones.
