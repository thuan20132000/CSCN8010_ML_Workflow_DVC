# ML Workflow with DVC

A reproducible machine learning pipeline for MNIST digit classification using [DVC](https://dvc.org/) (Data Version Control) and PyTorch. This project demonstrates how to build structured, traceable, and collaborative ML workflows by combining Git for code versioning with DVC for data and model tracking.

Large artifacts (datasets, model weights, predictions) are stored in **Google Drive** via the DVC remote. Only lightweight metadata and pointers are committed to Git.

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
├── .dvc/
│   ├── config             # DVC remote URL and client ID (committed to Git)
│   └── config.local       # OAuth client secret (gitignored, local only)
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
- 1 convolutional layer (1 → 8 channels, 3×3 kernel)
- Max pooling (2×2)
- 1 fully connected layer (8×13×13 → 10 classes)

## Getting Started

### Prerequisites

- Python 3.8+
- Git
- A Google account with access to the shared Google Drive folder

### Installation

```bash
git clone <repo-url>
cd ML_Workflow_DVC
pip install "dvc[gdrive]" torch torchvision scikit-learn pandas pyyaml
```

### Configure the DVC Remote

The remote URL and OAuth client ID are already committed in `.dvc/config`. You only need to supply your own client secret locally (it is never committed):

```bash
dvc remote modify --local myremote gdrive_client_secret 'YOUR_CLIENT_SECRET'
```

> To get a client secret, create an OAuth 2.0 Desktop app credential in [Google Cloud Console](https://console.cloud.google.com) → APIs & Services → Credentials, then add your Google account as a test user in the OAuth consent screen.

### Pull Artifacts and Run the Pipeline

```bash
dvc pull        # Download datasets, model, and predictions from Google Drive
dvc repro       # Re-run any stages whose inputs have changed
dvc metrics show
```

`dvc pull` will open a browser window for Google OAuth on the first run. Subsequent runs use a cached token.

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

Git stores code, pipeline definitions, parameters, and DVC metadata pointers. DVC remote storage (Google Drive) stores the actual binary artifacts.

### Push changes

```bash
# Stage and commit code + DVC metadata (includes .dvc/config with remote URL)
git add .dvc/config dvc.yaml dvc.lock params.yaml src/ metrics.json predictions.json
git commit -m "Update experiment"
git push

# Push large artifacts to Google Drive
dvc push
```

### Reproduce on another machine

```bash
git clone <repo-url>
cd ML_Workflow_DVC
pip install "dvc[gdrive]" torch torchvision scikit-learn pandas pyyaml

# Supply the client secret locally (not stored in Git)
dvc remote modify --local myremote gdrive_client_secret 'YOUR_CLIENT_SECRET'

dvc pull        # Download artifacts from Google Drive (OAuth browser login on first run)
dvc repro       # Re-run pipeline (skips stages if artifacts are up to date)
dvc metrics show
```

## DVC Remote Configuration

The remote is pre-configured in `.dvc/config`:

```ini
[core]
    remote = myremote
['remote "myremote"']
    url = gdrive://1iA_-KWT_le5l9PpBDr4o7nIOhIFTuf3_
    gdrive_client_id = <client-id>
```

The `gdrive_client_secret` is stored in `.dvc/config.local` (gitignored). Every collaborator must set it once with `dvc remote modify --local`.

### Common Setup Errors

| Error | Fix |
|-------|-----|
| `No module named 'dvc_gdrive'` | `pip install "dvc[gdrive]"` |
| `This app is blocked` | Use your own OAuth credentials instead of DVC's built-in app |
| `Error 403: access_denied` | Add your Google account as a test user in the OAuth consent screen |
| `Expected credentials type 'service_account'` | Remove `gdrive_use_service_account` setting; use `gdrive_client_id` + `gdrive_client_secret` instead |

## Key DVC Commands

| Command | Description |
|---------|-------------|
| `dvc init` | Initialize DVC in a Git repo |
| `dvc repro` | Reproduce the pipeline |
| `dvc metrics show` | Display current metrics |
| `dvc metrics diff` | Compare metrics between experiments |
| `dvc push` | Push tracked artifacts to Google Drive |
| `dvc pull` | Pull tracked artifacts from Google Drive |
| `dvc status -c` | Compare local cache against the remote |
| `dvc dag` | Visualize the pipeline DAG |
