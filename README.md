# Baseball Strike Zone Classification with SVM

This project builds and visualizes Support Vector Machine (SVM) models to classify baseball pitches as **strike** or **ball** using pitch location data.

The main workflow is implemented in a Jupyter notebook and compares multiple MLB players:
- Aaron Judge
- Jose Altuve
- David Ortiz

---

## Project Overview

The notebook demonstrates an end-to-end machine learning flow:
1. Load player pitch datasets
2. Clean and preprocess labels/features
3. Train an RBF-kernel SVM
4. Visualize decision boundaries
5. Tune hyperparameters (`gamma`, `C`)
6. Compare performance across players
7. Improve the model with an extra feature (`strikes`)

---

## Repository Structure

- `script.ipynb` — Main notebook with data analysis, training, tuning, and plots
- `players.py` — Loads CSV datasets into pandas DataFrames
- `svm_visualization.py` — Helper functions for plotting SVM decision boundaries
- `aaron_judge.csv`, `jose_altuve.csv`, `david_ortiz.csv` — Pitch-level datasets

---

## Data Source

The pitch data used in this project was collected using the Python package `pybaseball`.

- Data source package: `pybaseball`
- Export format used in this repo: CSV files per player

---

## Requirements

- Python 3.9+
- Jupyter Notebook (or JupyterLab)
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `pybaseball`

Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn pybaseball jupyter
```

---

## How to Run

1. Make sure the CSV files are present in the project root:
   - `aaron_judge.csv`
   - `jose_altuve.csv`
   - `david_ortiz.csv`
2. Open the notebook:

```bash
jupyter notebook script.ipynb
```

3. Run cells from top to bottom.

---

## Key Modeling Details

- Target label mapping:
  - `S` → `1` (Strike)
  - `B` → `0` (Ball)
- Base features:
  - `plate_x`
  - `plate_z`
- Additional feature in later experiment:
  - `strikes`
- Model:
  - `sklearn.svm.SVC` with RBF kernel
- Validation:
  - `train_test_split(..., random_state=1)`

---

## Notes

- The notebook includes visual diagnostics for underfitting/overfitting behavior as `gamma` and `C` change.
- Utility plotting in `svm_visualization.py` assumes a trained classifier and an axis with limits already set.
- `.gitignore` excludes CSV files and `__pycache__/`.

---

## Next Improvements (Optional)

- Replace manual hyperparameter loops with `GridSearchCV`
- Add cross-validation metrics (precision, recall, F1)
- Save trained models with `joblib`
- Add reproducible scripts for non-notebook execution
