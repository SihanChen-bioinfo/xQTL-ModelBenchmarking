# GWAVA for Functional xQTL Prediction

This directory contains scripts for functional xQTL prediction using GWAVA-derived genomic annotations. The framework extends the original GWAVA annotation strategy and applies machine learning models to predict functional variants across multiple xQTL types.

## Overview

The workflow consists of two steps:

1. Annotate genomic variants using GWAVA.
2. Train and evaluate xQTL prediction models using the generated annotations.

---

## Step 1: Generate GWAVA Annotations

Variants should first be annotated using the GWAVA annotation pipeline.

For detailed installation instructions and annotation resources, please refer to the original GWAVA repository:

https://github.com/wenweiliang/GWAVA

The input is typically a BED file containing variant coordinates. GWAVA generates a feature matrix that can be used for downstream machine learning analyses.

---

## Step 2: Train an xQTL Prediction Model

After generating GWAVA annotations, use `gwava_predict_xQTL.py` to train and evaluate predictive models.

### Example

```bash
python gwava_predict_xQTL.py \
    --mode train \
    --model xgb \
    --auto_tune \
    --folds 5 \
    --data_subdir all_xQTL_data \
    --dataset apaQTL \
    --out_dir apaQTL_xgb
```

### Parameters

| Parameter | Description |
|------------|------------|
| `--mode` | Running mode. Choose from `train` or `predict`. Default: `train`. |
| `--model` | Machine learning model to use. Supported options: `rf`, `xgb`, `l1`, and `enet`. Default: `xgb`. |
| `--gwava_dir` | Path to the GWAVA installation directory. Default: current directory (`.`) or the value of the `GWAVA_DIR` environment variable. |
| `--data_subdir` | Subdirectory containing GWAVA-annotated datasets. Default: `xQTL_data`. |
| `--dataset` | Dataset name (required), such as `eQTL`, `mQTL`, `caQTL`, or `apaQTL`. |
| `--out_dir` | Output directory. All generated files, including trained models and evaluation results, are written directly to this directory. |
| `--model_dir` | Directory containing trained models for prediction mode. If not specified, `--out_dir` is used. Used only in `predict` mode. |
| `--predict_out` | Output file for prediction results. Used only in `predict` mode. |
| `--folds` | Number of chromosome-based folds used for hyperparameter tuning on the training set. Default: `5`. |
| `--random_state` | Random seed used for data splitting and model training. Default: `0`. |
| `--n_jobs` | Number of CPU cores used for parallel computation. Use `-1` to utilize all available cores. Default: `-1`. |
| `--auto_tune` | Enable automatic hyperparameter tuning using chromosome-based cross-validation on the training set (default behavior). |
| `--no_auto_tune` | Disable automatic hyperparameter tuning and train the model using predefined default hyperparameters. |
---

## Output

The output directory contains model training and evaluation results, including:

- Trained model files
- Performance metrics (AUROC, AUPRC, MCC, etc.)
- Feature importance scores (when supported by the selected model)
- Prediction results (in predict mode)




