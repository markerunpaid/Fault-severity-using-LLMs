# Fault Severity Prediction

A machine learning project for predicting bug/fault severity using code models and classical machine learning classifiers.

## Dataset

The dataset used in this project is sourced from:

**Source:** https://github.com/EhsanMashhadi/ISSRE2023-BugSeverityPrediction/tree/main/experiments/data

This dataset contains bug reports with associated code repositories and severity labels, enabling research on automated bug severity classification.

## Project Structure

```
fault-severity/
├── checkpoints/              # Trained model checkpoints (.pt files)
│   ├── best_codebert.pt
│   ├── best_codet5p.pt
│   ├── best_graphcodebert.pt
│   ├── best_unixcoder.pt
│   └── ablation/             # Models for ablation studies
├── data/                      # Dataset files
│   ├── train_final.csv        # Final training dataset
│   ├── test_final.csv         # Final test dataset
│   ├── train_raw.csv          # Raw training data
│   ├── test_raw.csv           # Raw test data
│   ├── train_with_metrics.csv # Training data with extracted metrics
│   ├── test_with_metrics.csv  # Test data with extracted metrics
│   └── train_smote.csv        # SMOTE-augmented training data
├── embeddings/                # Pre-computed embeddings
│   ├── codebert_train_embeddings.npy
│   ├── codebert_test_embeddings.npy
│   ├── codet5p_train_embeddings.npy
│   ├── codet5p_test_embeddings.npy
│   ├── graphcodebert_train_embeddings.npy
│   ├── graphcodebert_test_embeddings.npy
│   ├── unixcoder_train_embeddings.npy
│   └── unixcoder_test_embeddings.npy
├── ml classifiers/            # Classical ML classifier results
│   ├── ml_classifiers_comparison.csv
│   └── ml_classifiers_results.json
├── results/                   # Training and evaluation results
│   ├── codebert_results.json
│   ├── codet5p_results.json
│   ├── graphcodebert_results.json
│   ├── unixcoder_results.json
│   ├── *_best_params.json     # Hyperparameter tuning results
│   ├── *_smote_results.json   # SMOTE variant results
│   └── full_evaluation.json   # Comprehensive evaluation metrics
└── src/                       # Source code
    ├── step1_preprocess.py           # Data preprocessing
    ├── step2_extract_metrics.py      # Extract code metrics
    ├── step3_scale.py                # Data scaling
    ├── apply_smote.py                # SMOTE data augmentation
    ├── extract_embeddings.py         # Extract model embeddings
    ├── tune_and_train.py             # Hyperparameter tuning and training
    ├── train_codebert_full.py        # CodeBERT training
    ├── train_smote_only.py           # SMOTE variant training
    ├── train_unixcoder_classical.py  # Classical ML classifiers
    ├── full_evaluation.py            # Comprehensive model evaluation
    ├── model.py                      # Model architecture
    ├── trainer.py                    # Training loop utilities
    ├── dataset.py                    # Dataset handling
    ├── preprocessing.py              # Data preprocessing utilities
    ├── metrics_extractor.py          # Code metrics extraction
    └── ablation/                     # Ablation study scripts
```

## Models

This project implements and evaluates the following code-based models:

- **CodeBERT**: Pre-trained BERT model for code understanding
- **CodeT5+**: Encoder-decoder model for code tasks
- **GraphCodeBERT**: Graph-aware BERT for code understanding
- **UniXcoder**: Unified cross-lingual code pre-training model

Additionally, classical machine learning classifiers are trained on extracted features for comparison.

## Key Features

- **Multi-model architecture**: Support for multiple pre-trained code models
- **Feature engineering**: Extraction of code metrics (complexity, coupling, etc.)
- **Data augmentation**: SMOTE (Synthetic Minority Over-sampling Technique) for handling class imbalance
- **Hyperparameter tuning**: Grid/random search for optimal model parameters
- **Ablation studies**: Systematic evaluation of different components (scalers, metrics, SMOTE, etc.)
- **Comprehensive evaluation**: Multiple metrics and detailed result analysis

## Workflow

1. **Preprocessing** (`step1_preprocess.py`): Clean and prepare raw data
2. **Metrics Extraction** (`step2_extract_metrics.py`): Extract code metrics from bug reports
3. **Scaling** (`step3_scale.py`): Normalize features
4. **SMOTE Augmentation** (`apply_smote.py`): Address class imbalance
5. **Embedding Extraction** (`extract_embeddings.py`): Generate embeddings using pre-trained models
6. **Model Training** (`tune_and_train.py`): Train and tune models with hyperparameter search
7. **Evaluation** (`full_evaluation.py`): Comprehensive evaluation across all models

## Results

Results are organized by model and variant:
- **Full models**: Trained on complete feature sets
- **SMOTE variants**: Models trained on SMOTE-augmented data
- **Ablation studies**: Models evaluating specific components

Detailed results are available in the `results/` directory in JSON format.

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Scikit-learn
- Imbalanced-learn (imblearn)
- NumPy
- Pandas
- XGBoost
- CatBoost
- LightGBM

## License

Please refer to the original dataset repository for licensing information.

## Citation

If you use this dataset or project, please cite the original work:

```
https://github.com/EhsanMashhadi/ISSRE2023-BugSeverityPrediction
```
