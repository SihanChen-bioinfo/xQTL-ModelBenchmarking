# Enformer for Functional xQTL Prediction

This directory contains scripts for functional xQTL prediction using genome-wide regulatory feature predictions derived from Enformer. The workflow first generates variant-level regulatory scores using Enformer, and then trains machine learning models to predict functional xQTL variants.

---

## Overview

The workflow consists of two steps:

1. Predict regulatory chromatin features using Enformer.
2. Train and evaluate xQTL prediction models using Enformer-derived features.

---

## Step 1: Generate Enformer Features

Enformer is a deep learning model for predicting gene regulation from DNA sequence, capturing long-range chromatin and transcriptional regulatory effects.

For the original implementation, see:

https://github.com/FunctionLab/ExPecto

### Example

```bash
python runEnformer.py \
    --vcf input.vcf \
    --fasta hg38.fa.gz \
    --mode full \
    --out output.score.txt
```

### Output

The output file (`output.score.txt`) contains variant-level regulatory feature scores across multiple Enformer tracks, representing predicted chromatin and transcriptional activity changes.

These features are used as input for downstream xQTL classification.

---

## Step 2: Train an xQTL Prediction Model

After generating Enformer features, use `Enformer_predict_xQTL.py` to train and evaluate models.

### Example

```bash
python Enformer_predict_xQTL.py \
    --mode train \
    --model xgb \
    --auto_tune \
    --folds 5 \
    --enformer_csv score.txt \
    --labels_csv labels.txt \
    --out_dir out_dir
```

---

## Parameters

| Parameter | Description |
|------------|------------|
| `--mode` | Running mode. Choose from `train` or `predict`. Default: `train`. |
| `--enformer_csv` | Input CSV generated from Enformer scoring pipeline. Contains variant IDs and regulatory track features. Required. |
| `--labels_csv` | Label file containing variant IDs and binary labels (0/1). Required for training mode. |
| `--id_col` | Column name for variant identifiers. Default: `id`. |
| `--label_col` | Column name for binary labels. Default: `label`. |
| `--folds` | Number of chromosome-based folds used for hyperparameter tuning on training data. Default: `5`. |
| `--n_jobs` | Number of parallel CPU jobs (used in RandomForest). Default: `-1`. |
| `--out_dir` | Output directory for models, evaluation results, and predictions. Required. |
| `--model` | Machine learning model type. Supported options: `rf` (Random Forest), `xgb` (XGBoost), `l1` (L1 logistic regression), `enet` (Elastic Net logistic regression). |
| `--auto_tune` | Enable automatic hyperparameter tuning using chromosome-based cross-validation on training data. |
| `--no_auto_tune` | Disable hyperparameter tuning and use default model settings. |
| `--random_state` | Random seed for reproducibility. Default: `0`. |
| `--chrom_col` | Chromosome column name used for chromosome-based splitting. Default: `chrom`. |
| `--model_dir` | Directory containing trained models for prediction mode. Must match training model type. |
| `--predict_out` | Output filename for prediction results. Default: `predictions_{model}.csv`. |

---

## Output

The output directory contains:

- Trained models
- Performance metrics (AUROC, AUPRC, MCC, etc.)
- Feature importance scores (if supported)
- Prediction results (if in predict mode)

