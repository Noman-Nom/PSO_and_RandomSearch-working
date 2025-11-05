# Hyperparameter Optimization Research - Experiment Baseline

## Overview
This document describes the baseline methodology for comparing global optimization algorithms (Random Search and PSO) for neural network hyperparameter tuning.

## Baseline Methodology

### Two-Phase Training Pipeline

#### Phase 1: Exploration (5,000 total epochs)
- **Goal**: Find best hyperparameters using optimization algorithms
- **Budget**: Fixed at 5,000 total training epochs across all evaluations
- **Purpose**: Fair comparison under fixed compute budget

#### Phase 2: Exploitation (15,000-20,000 epochs)
- **Goal**: Retrain best configuration for maximum performance
- **Budget**: 15,000-20,000 epochs for final model training
- **Purpose**: Achieve best possible performance with optimal hyperparameters

### Configuration Details
--
#### Random Search
- **Iterations**: 25 trials
- **Epochs per trial**: 200 epochs
- **Total exploration epochs**: 25 × 200 = **5,000 epochs** ✅
- **Early stopping**: Enabled (patience: 20)

#### Particle Swarm Optimization (PSO)
- **Iterations**: 10 iterations
- **Population size**: 10 particles
- **Epochs per particle**: 50 epochs
- **Total exploration epochs**: 10 × 10 × 50 = **5,000 epochs** ✅
- **PSO parameters**:
  - Inertia (w): 0.7
  - Cognitive (c1): 1.5
  - Social (c2): 1.5
- **Early stopping**: Enabled (patience: 10)

### Dataset and Model

#### Dataset
- **Primary**: Full MNIST (10 classes)
- **Alternative**: FashionMNIST or CIFAR-10
- **Split**: Train/Val/Test with 10% validation, 10% test

#### Model Architecture
- **Type**: Modern MLP with:
  - Batch Normalization
  - Dropout regularization
  - Multiple hidden layers (1-3 layers)
  - Proper weight initialization (He initialization)
  - ReLU activations

#### Search Space
- **Learning rate**: 0.0001 - 0.01 (log scale)
- **Batch size**: 32 - 256 (log scale)
- **Hidden size**: 64 - 512 (linear scale)
- **Number of layers**: 1 - 3 (linear scale)
- **Dropout**: 0.1 - 0.5 (linear scale)
- **Weight decay**: 0.00001 - 0.001 (log scale)

### Training Configuration

#### Exploration Phase
- **Optimizer**: Adam
- **Early stopping**: Enabled to prevent overfitting during exploration
- **Validation**: Uses validation set to select best hyperparameters

#### Exploitation Phase
- **Max epochs**: 15,000 (can be increased to 20,000)
- **Early stopping patience**: 50 epochs
- **Final evaluation**: Test set accuracy reported

### Metrics and Reporting

#### Saved Artifacts
1. **Model checkpoints**: Best model state dict (.pth files)
2. **Training history**: Loss and accuracy curves (JSON + plots)
3. **Optimization history**: All evaluations during exploration (JSON)
4. **Summary statistics**: Mean, std, best, worst across multiple runs

#### Performance Metrics
- **Validation accuracy**: Used during exploration to select best config
- **Test accuracy**: Final performance metric for comparison
- **Training time**: Wall-clock time for reproducibility
- **Convergence curves**: Plots showing optimization progress

### Reproducibility

#### Random Seeds
- **Base seed**: 42
- **Multiple runs**: 3 runs per optimizer for statistical significance
- **Seed per run**: base_seed + run_number

#### Fixed Budget Comparison
Both optimizers use exactly **5,000 total epochs** for exploration, ensuring:
- Fair comparison under same compute budget
- No advantage from additional training
- Clear baseline for publication

### Expected Results Format

For publication, results should include:
1. **Table**: Comparison of Random Search vs PSO
   - Mean ± Std test accuracy
   - Best configuration found
   - Total time taken

2. **Plots**:
   - Convergence curves (validation accuracy vs evaluations)
   - Training curves (loss/accuracy vs epochs)
   - Hyperparameter distributions

3. **Analysis**:
   - Which optimizer found better hyperparameters?
   - Convergence speed comparison
   - Final performance comparison

### Running Experiments

```bash
cd experiments
python run_experiment.py
```

The configuration is in `config.yaml`. Key settings:
- Enable/disable optimizers
- Adjust epoch budgets (maintain 5,000 total for exploration)
- Select dataset (MNIST, FashionMNIST, CIFAR-10)
- Configure model architecture

### Notes

- The 5,000 epoch budget is **strict** - this ensures fair comparison
- Early stopping may reduce actual epochs used, but budget is fixed per evaluation
- Results directory structure: `results/DATASET_MODEL_TIMESTAMP/`
- Each optimizer run saved in: `optimizer_name/run_SEED/`

