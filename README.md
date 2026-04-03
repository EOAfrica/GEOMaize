# GEOMaize — Few-Shot Maize Yield Prediction in Ghana using Presto

GEOMaize is a research repository that explores **maize yield prediction in Ghana** using satellite Earth Observation (EO) data and the **Presto** foundation model (Pretrained Remote Sensing Transformer). The project investigates both **continuous yield regression** and **binary yield classification** (Low / High) in a few-shot learning regime, and provides a full end-to-end inference pipeline to generate spatially explicit yield maps.

The work demonstrates that binary Low/High classification using fine-tuned Presto — even with a small number of training samples (~160 field-year records) — produces robust, operationally meaningful predictions from freely available Sentinel-2, Sentinel-1, and auxiliary data.

---

## Repository Structure

```
GEOMaize/
├── notebooks/                  # All Jupyter notebooks (see below)
│   ├── GEOMaize_maize_yield_prediction.ipynb   # Main end-to-end pipeline (start here)
│   ├── GEOMaize_FSL_pipeline.ipynb             # Few-shot learning pipeline overview
│   ├── geomaize_experiments_binary.ipynb       # Binary classification experiments
│   ├── geomaize_experiments_regression.ipynb   # Regression experiments
│   └── demo_utils.py                           # Shared plotting and utility functions
│
├── data/
│   ├── datasets/               # Parquet files with extracted field-level features
│   ├── inference/              # Pre-extracted NetCDF inference cubes and masks
│   └── input_data/             # Raw or intermediate inputs
│
├── models/                     # Saved fine-tuned Presto model checkpoints
├── requirements.txt            # Python dependencies
└── workflow/                   # Supporting workflow scripts
```

---

## Notebooks Overview

### 1. `GEOMaize_maize_yield_prediction.ipynb` — *End-to-end pipeline*
The **main end-to-end notebook** for demonstrating the full pipeline:
- Loads and explores yield field data (157 field-year records, Ghana 2021–2025)
- Binarizes yield at the 1220 kg/H threshold (Low / High classes)
- Fine-tunes Presto for binary classification on train/val/test splits
- Evaluates model performance via confusion matrices and yield scatter plots
- Runs inference on a 10 km × 10 km area around Tamale producing a pixel-level yield probability map (GeoTIFF)

This is the recommended entry point for anyone wanting to understand the approach or reproduce the results.

---

### 2. `geomaize_experiments_binary.ipynb` — *Classification experiments*
Compares four binary classification approaches side-by-side:
- **Random Forest** — tabular baseline with validation-based early stopping
- **CatBoost** — gradient boosting with early stopping on validation loss
- **Presto (fine-tuned)** — end-to-end fine-tuning with BCEWithLogitsLoss
- **Presto + CatBoost** — frozen Presto encoder → CatBoost downstream classifier

Produces accuracy/F1 bar charts, a 4×3 confusion matrix grid, and per-sample yield scatter plots coloured by classification correctness.

**Key result**: Fine-tuned Presto achieves the best overall generalization (test accuracy 0.88 in Presto+CatBoost configuration; most stable validation behaviour with fine-tuned Presto alone).

---

### 3. `geomaize_experiments_regression.ipynb` — *Regression experiments*
Explores direct continuous yield regression with the same four model families (Random Forest Regressor, CatBoost Regressor, Presto fine-tuned with MSELoss, Presto+CatBoost Regressor).

Produces predicted-vs-true yield scatter plots and a grouped R² bar chart across train/val/test splits.

**Key result**: All models show strong overfitting (train R² 0.74–0.80 vs. validation R² near zero). Best test R² is 0.40 (fine-tuned Presto). Continuous yield prediction is insufficiently reliable in this data regime, motivating the shift to binary classification.

---

### 4. `GEOMaize_FSL_pipeline.ipynb` — *Few-shot learning pipeline*
Provides an overarching view of the few-shot learning (FSL) methodology used throughout the project, including feature extraction, dataset construction, and Presto integration.

---

## Setup and Installation

All required dependencies are managed through the `scaleag-vito` package, which bundles dataset wrappers, Presto utilities, OpenEO extraction tools, and all transitive dependencies needed to run the notebooks.

---

### Option A — Terrascope Hub (recommended, no local setup required)

The easiest way to run these notebooks is via the **[Terrascope Hub](https://hub.terrascope.be/en)**:

1. register as a new user on Terrascope or use one of the supported EGI eduGAIN login methods to get started
2. Once you are prompted with "Server Options", make sure to select the "Worldcereal" image. Did you choose "Terrascope" by accident? Then go to File > Hub Control Panel > Stop my server, and click the link below once again

<a href="https://notebooks.terrascope.be/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FEOAfrica%2FGEOMaize&urlpath=lab%2Ftree%2FGEOMaize%2Fnotebooks%2FGEOMaize_maize_yield_prediction.ipynb&branch=main"><img src="https://img.shields.io/badge/run%20end--to--end%20pipeline-Terrascope-brightgreen" alt="Run end-to-end pipeline" valign="middle"></a>

<a href="https://notebooks.terrascope.be/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FEOAfrica%2FGEOMaize&urlpath=lab%2Ftree%2FGEOMaize%2Fnotebooks%2Fgeomaize_experiments_binary.ipynb&branch=main"><img src="https://img.shields.io/badge/run%20binary%20classification-Terrascope-brightgreen" alt="Run binary classification experiments" valign="middle"></a>

<a href="https://notebooks.terrascope.be/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FEOAfrica%2FGEOMaize&urlpath=lab%2Ftree%2FGEOMaize%2Fnotebooks%2Fgeomaize_experiments_regression.ipynb&branch=main"><img src="https://img.shields.io/badge/run%20regression%20experiments-Terrascope-brightgreen" alt="Run regression experiments" valign="middle"></a>
---

### Option B — Local pip environment

```bash
# 1. Create a virtual environment
python -m venv --system-site-packages .venv

# 2. Activate it
source .venv/bin/activate

# 3. Install all dependencies via the scaleag-vito package
pip install git+https://github.com/ScaleAGData/scaleag-vito.git@prometheo-integration

# 4. Register the kernel for Jupyter
ipython kernel install --user --name=geomaize
```

## Data

- **Field dataset**: `data/datasets/ghana_maize_multiyear_extractions.parquet` — pre-extracted Sentinel-2, Sentinel-1, DEM, and METEO features for each field-year record, with train/val/test split labels and binarized yield target (`bin` column).
- **Inference data**: `data/inference/month/ref_id=inference_extent_10km_latlon/` — pre-extracted monthly composite NetCDF for a 10 km × 10 km area around Tamale, plus a binary maize mask GeoTIFF.

Large NetCDF files (> 50 MB) are excluded from git history. If running inference, ensure the relevant `.nc` file is present in the expected path.

---

## Key Dependencies

| Package | Role |
|---|---|
| `prometheo` | Presto foundation model (fine-tuning, inference) |
| `scaleagdata-vito` | Dataset wrappers, inference utilities, OpenEO extraction |
| `catboost` | Gradient boosting classifier/regressor |
| `scikit-learn` | Random Forest, pipelines, metrics |
| `xarray` / `rioxarray` | Geospatial raster handling |
| `geopandas` | Vector data and spatial extents |
| `torch` | Deep learning backend for Presto |


