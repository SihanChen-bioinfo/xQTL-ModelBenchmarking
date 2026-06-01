# Benchmarking Deep Learning Models for xQTL Prediction

A benchmarking framework for evaluating deep learning and machine learning models in xQTL prediction tasks.

This repository contains implementations for extracting chromatin feature predictions from multiple genomic sequence models and benchmarking their performance in downstream xQTL prediction.

## Repository Structure

```text

.

├── AlphaGenome/
│    ├── runAlphaGenome.py
│    └── AlphaGenome_predict_xQTL.py
│ 
├── Borzoi/
│    └── Borzoi_predict_xQTL.py
│ 
├── DeepSEA/
│    ├── runDeepSEA.py
│    └── DeepSEA_predict_xQTL.py
│ 
├── Enformer/
│    ├── runEnformer.py
│    └── Enformer_predict_xQTL.py
│ 
├── ExPecto/
│    ├── runExPecto.py
│    └── ExPecto_predict_xQTL.py
│ 
├── GWAVA/
│    └──  GWAVA_predict_xQTL.py
│ 
└── feature_enrichment.py

```

## Model Directories

Each model directory contains two main scripts:

### 1. Chromatin Feature Prediction

Generates regulatory feature predictions from genomic sequences.

Predicted features may include:

- Chromatin accessibility

- Histone modifications

- Transcription factor binding

- Gene expression signals

- Other model-specific regulatory outputs

### 2. xQTL Prediction

Uses predicted chromatin features as input features for xQTL classification.

Supported machine learning algorithms include:

- Random Forest (RF)

- XGBoost

- Logistic Regression (L1/LASSO)

- Elastic Net

- Support Vector Machine (SVM)



## Feature Enrichment Analysis

### feature_enrichment.py

This script performs feature enrichment analyses using chromatin features identified as important for xQTL prediction.

### Usage

```bash

python feature_enrichment.py \

    feature_importance.txt \

    feature_to_tissue.txt \

    output_dir

```

### Input Files

#### feature_importance.txt

Feature importance scores generated from xQTL prediction models.

Example:

```text

feature_name    importance

DNASE:K562      0.084

H3K27ac:GM12878 0.072

...

```

#### feature_to_tissue.txt

Annotation file mapping chromatin features to assays and tissues.

Example:

```text

feature_name    assay       tissue

DNASE:K562      DNase-seq   Blood

H3K27ac:GM12878 H3K27ac     Blood

...

```

### Output

Results will be written to:

```text

output_dir/
│
├── feature_enrichment_assay.tsv
│
└── feature_enrichment_tissue.tsv

```


---

## Reproducibility

This repository contains all scripts used to generate the results reported in the manuscript, including chromatin feature prediction, xQTL classification, model benchmarking, and feature enrichment analyses.
