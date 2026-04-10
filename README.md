# ML Workflow with DVC

A reproducible machine learning pipeline for MNIST digit classification using [DVC](https://dvc.org/) (Data Version Control) and PyTorch. This project demonstrates how to build structured, traceable, and collaborative ML workflows by combining Git for code versioning with DVC for data and model tracking.

## Project Structure

```
ML_Workflow_DVC/
├── data/
│   ├── raw/               # Raw MNIST download (DVC-tracked)
│   └── processed/         # Preprocessed tensors (DVC-tracked)
├── src/
│   ├── prepare.py         # Download and preprocess MNIST data
│   ├── train.py           # Train a simple CNN model
│   └── predict.py         # Run inference on test data
├── params.yaml            # Hyperparameters (epochs, lr, batch_size)
├── dvc.yaml               # Pipeline stage definitions
├── dvc.lock               # Pipeline state and file hashes
├── model.pt               # Trained model weights (DVC-tracked)
├── metrics.json           # Training evaluation metrics
└── predictions.json       # Prediction results on test set
```

## Pipeline Overview

The DVC pipeline consists of three stages:

```
prepare --> train --> predict
```

| Stage | Script | Inputs | Outputs |
|-------|--------|--------|---------|
| **prepare** | `src/prepare.py` | MNIST download | `data/processed/train.pt`, `data/processed/test.pt` |
| **train** | `src/train.py` | `data/processed/`, `params.yaml` | `model.pt`, `metrics.json` |
| **predict** | `src/predict.py` | `model.pt`, `data/processed/test.pt` | `predictions.json` |

### Model

A simple CNN architecture:
- 1 convolutional layer (1 -> 8 channels, 3x3 kernel)
- Max pooling (2x2)
- 1 fully connected layer (8x13x13 -> 10 classes)

## Getting Started

### Prerequisites

- Python 3.8+
- Git

### Installation

```bash
git clone <repo-url>
cd ML_Workflow_DVC
pip install dvc torch torchvision scikit-learn pandas pyyaml
```

### Run the Pipeline

```bash
dvc repro
```

This executes all pipeline stages in dependency order. DVC skips stages whose inputs haven't changed.

### View Metrics

```bash
dvc metrics show
```

## Experimenting with Hyperparameters

Modify `params.yaml` to change training configuration:

```yaml
epochs: 5
lr: 0.001
batch_size: 64
```

Then re-run the pipeline and compare results:

```bash
dvc repro
dvc metrics diff
```

DVC automatically detects which stages are affected by parameter changes and only re-runs those stages.

## Collaboration Workflow

Git tracks code, configuration, and DVC metadata. DVC tracks large files (datasets, models).

### Push changes

```bash
git add .
git commit -m "Update experiment"
git push

dvc push    # Pushes data/model artifacts to remote storage
```

### Reproduce on another machine

```bash
git clone <repo-url>
cd ML_Workflow_DVC
pip install dvc torch torchvision scikit-learn pandas pyyaml
dvc pull    # Pull data/model artifacts from remote storage
dvc repro   # Re-run pipeline (skips stages if artifacts are up to date)
```

## Key DVC Commands

| Command | Description |
|---------|-------------|
| `dvc init` | Initialize DVC in a Git repo |
| `dvc repro` | Reproduce the pipeline |
| `dvc metrics show` | Display current metrics |
| `dvc metrics diff` | Compare metrics between experiments |
| `dvc push` | Push tracked artifacts to remote storage |
| `dvc pull` | Pull tracked artifacts from remote storage |
| `dvc dag` | Visualize the pipeline DAG |
