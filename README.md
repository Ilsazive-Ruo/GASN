# DrugCombination Scripts

This directory contains scripts for drug-combination classification, feature comparison, cold-start evaluation, similarity analysis, and prediction with saved 10-fold models.

## Environment

Main package versions are listed in `requirements_scripts.txt`.

## Data

Default input files are expected under:

```text
data/antis_f.csv
data/harms_f.csv
data/others_f.csv
data/syn_f.csv
data/ToBeScreened.csv
```

Training CSV files should contain:

```text
id-1,name-1,atc-1,smiles-1,id-2,name-2,atc-2,smiles-2,class
```

Prediction CSV files should contain either `id-1,id-2` or `smiles-1,smiles-2`.


Feature types:

- `des`: RDKit molecular descriptors
- `fp`: Morgan fingerprints
- `des_fp`: Morgan fingerprints plus RDKit descriptors

## Scripts

### 1. Compare XGBoost Features

Runs stratified cross-validation for `des`, `fp`, and `des_fp` by default.

```powershell
python compare_xgb_features.py
```

Outputs include:

```text
xgb_feature_compare/feature_compare_metrics.csv
xgb_feature_compare/feature_compare_summary.csv
xgb_feature_compare/curves/
```

### 2. Cross-Validate Models

Compares all configured models in `dcp_common.py`.

```powershell
python cross_validate.py
```

Outputs include metrics, confusion matrices, and ROC/PR/calibration curves.

### 3. Cold-Start Evaluation

Evaluates drug-level cold-start performance.

```powershell
python xgb_cold_start.py --feat des --cold_start_mode double
```

Modes:

- `any`: at least one held-out drug
- `single`: exactly one held-out drug
- `double`: both drugs held out

### 4. XGBoost Grid Search

Runs `GridSearchCV` for XGBoost.

```powershell
python grid_search_xgb.py --feat des --cv 10
```

Outputs:

```text
scripts_outputs/grid_search_xgb/des/cv_results.csv
scripts_outputs/grid_search_xgb/des/best_params.json
scripts_outputs/grid_search_xgb/des/best_model/
```

### 5. Similarity Matrices

Computes Tanimoto similarity and descriptor Euclidean distance between `ToBeScreened.csv` molecules and dataset molecules.

```powershell
python compute_similarity_matrices.py
```

### 6. Predict With Saved 10-Fold Models

Uses models in `ten_fold_models/` to predict new pairs.

```powershell
python predict_10fold_models.py ^
  --model_dir ten_fold_models/xgb_des ^
  --input data/ToBeScreened.csv ^
  --output outputs/ten_fold_predictions.csv
```

The script saves all predictions and prints rows predicted as `synergistic`.

These include mean prediction score, SD, and per-fold scores.
