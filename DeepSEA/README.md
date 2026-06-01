# DeepSEA for Functional xQTL Prediction

This directory contains scripts for functional xQTL prediction using chromatin feature predictions derived from DeepSEA. The workflow first predicts variant-level chromatin effects using DeepSEA, and then trains machine learning models to classify functional xQTL variants.

---

## Overview

The workflow consists of two steps:

1. Predict chromatin features using DeepSEA.
2. Train and evaluate xQTL prediction models using DeepSEA-derived features.

---

## Step 1: Generate DeepSEA Chromatin Features

DeepSEA is a deep learning-based framework for predicting chromatin effects of sequence alterations, including transcription factor binding, DNase hypersensitivity, and histone modifications across multiple cell types.  
Official documentation and online tool are available at:

https://deepsea.princeton.edu/help/


### Example

```bash
python runDeepSEA.py input.vcf out_dir
```



---

## Step 2: Train an xQTL Prediction Model

After generating chromatin features, use `deepSEA_xQTL_CV10.py` to train and evaluate machine learning models.

### Example

```bash
python deepSEA_xQTL_CV10.py \
    --mode train \
    --model xgb \
    --auto_tune \
    --folds 5 \
    --features_csv input_chromatin_features.txt \
    --labels_csv labels.txt \
    --xqtl_type apaQTL \
    --out apaQTL_xgb
```

---

## Parameters

| Parameter | Description |
|------------|------------|
| `--features_csv` | Input feature file containing variant-level chromatin features. Format: `chr, pos, name, ref, alt, features...`. Required. |
| `--labels_csv` | Label file containing variant identifiers (`name`) and binary labels. Required for training mode. |
| `--xqtl_type` | Dataset tag used for naming output directories if `--out_dir` is not specified. Default: `deepsea_xqtl`. |
| `--mode` | Running mode. Choose from `train` or `predict`. Default: `train`. |
| `--model` | Machine learning model to use. Supported options: `xgb`, `rf`, `l1`, and `enet`. |
| `--n_jobs` | Number of CPU cores used for parallel computation. Default: `-1` (all cores). |
| `--auto_tune` | Enable automatic hyperparameter tuning using chromosome-based cross-validation on the training set. |
| `--no_auto_tune` | Disable hyperparameter tuning and use default model parameters. |
| `--folds` | Number of chromosome-based folds used for hyperparameter tuning. Default: `5`. |
| `--random_state` | Random seed for reproducibility. Default: `0`. |
| `--out_dir` | Output directory for results. Default: `{xqtl_type}_cross_validation_result`. |
| `--model_dir` | Directory containing pretrained models for prediction mode. Must match training model type. |
| `--predict_out` | Output filename for prediction results. Default: `predictions_{model}.csv`. |

---

## Output

The output directory contains:

- Trained models
- Validation results
- Feature importance
- Prediction results (in predict mode)



