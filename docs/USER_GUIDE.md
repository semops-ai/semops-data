# Data Systems Toolkit - User Guide

A practical guide to using the DevContainer environment for data science workflows.

## Quick Start

### 1. Open in DevContainer

```bash
# In VSCode:
# 1. Open this repo
# 2. Ctrl+Shift+P -> "Dev Containers: Reopen in Container"
# 3. Wait for build (~5 min first time)
```

### 2. Verify Environment

Run the test notebook to confirm everything works:

1. Open `notebooks/00-environment-test.ipynb` in VSCode
2. Click "Select Kernel" -> choose `/opt/conda/bin/python`
3. Run all cells with `Shift+Enter` or "Run All"

Or quick CLI check:
```bash
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

---

## Data Management

### Directory Structure

```
data-systems-toolkit/
├── data/                    # Your working datasets (gitignored)
│   ├── raw/                 # Original downloaded data
│   ├── processed/           # Cleaned/transformed data
│   └── interim/             # Intermediate processing steps
├── mlruns/                  # MLflow tracking (gitignored)
└── notebooks/               # Your analysis notebooks
```

### Where to Put Data

| Data Type | Location | Notes |
|-----------|----------|-------|
| Downloaded datasets | `data/raw/` | Original, never modify |
| Processed data | `data/processed/` | Cleaned, ready for modeling |
| Intermediate files | `data/interim/` | Temp processing outputs |
| HuggingFace cache | Auto-managed | `~/.cache/huggingface/` in container |

**First time setup:**
```bash
mkdir -p data/raw data/processed data/interim
```

---

## Acquiring Datasets

### Option 1: HuggingFace Datasets (Recommended)

Best for quick experiments - thousands of ready-to-use datasets.

```python
from datasets import load_dataset

# Load directly (downloads and caches automatically)
dataset = load_dataset("scikit-learn/iris")

# Load specific split
train = load_dataset("imdb", split="train")
test = load_dataset("imdb", split="test")

# Convert to pandas
df = dataset.to_pandas()
```

**Popular datasets for practice:**

| Dataset | Use Case | Load Command |
|---------|----------|--------------|
| Iris | Classification basics | `load_dataset("scikit-learn/iris")` |
| Wine | Multi-class classification | `load_dataset("scikit-learn/wine")` |
| California Housing | Regression | `load_dataset("scikit-learn/california-housing")` |
| IMDB Reviews | Text classification | `load_dataset("imdb")` |
| MNIST | Image classification | `load_dataset("mnist")` |

**Browse available datasets:** https://huggingface.co/datasets

### Option 2: Kaggle Datasets

```bash
# Install kaggle CLI (already in container)
pip install kaggle

# Configure API key (one-time setup)
# 1. Go to kaggle.com/account -> Create New API Token
# 2. Download kaggle.json
# 3. Place in container:
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download a dataset
kaggle datasets download -d <dataset-name> -p data/raw/
unzip data/raw/*.zip -d data/raw/
```

### Option 3: Direct Downloads

```python
import pandas as pd

# From URL
df = pd.read_csv("https://example.com/data.csv")
df.to_csv("data/raw/my_dataset.csv", index=False)

# From local file (mounted in container)
df = pd.read_csv("data/raw/my_local_file.csv")
```

### Option 4: Sklearn Built-in Datasets

```python
from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    fetch_california_housing,
    make_classification,  # Generate synthetic
    make_regression,
)

# Load as DataFrame
from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)
df = iris.frame
```

---

## MLflow Setup & Usage

### Initialize MLflow (First Time)

MLflow auto-initializes on first use. No setup required.

Tracking data saved to `mlruns/` (gitignored).

### Start the MLflow UI

```bash
# In container terminal:
mlflow ui --host 0.0.0.0

# Open the MLflow web interface in your browser
```

### Basic Experiment Tracking

```python
import mlflow

# Set experiment (creates if doesn't exist)
mlflow.set_experiment("my-experiment-name")

# Log a run
with mlflow.start_run(run_name="my-first-run"):
    # Log parameters
    mlflow.log_param("model_type", "random_forest")
    mlflow.log_param("n_estimators", 100)

    # Log metrics
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("f1_score", 0.93)

    # Log model (optional)
    mlflow.sklearn.log_model(model, "model")
```

### Compare Runs

In the MLflow UI:
1. Select experiments in left sidebar
2. Check runs to compare
3. Click "Compare" button
4. View metrics side-by-side

---

## Scikit-Learn Workflows

### Basic Classification Pipeline

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow

# 1. Load data
from datasets import load_dataset
dataset = load_dataset("scikit-learn/iris", split="train")
df = dataset.to_pandas()

# 2. Prepare features and target
X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = df["Species"]

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Scale features (optional but recommended)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train with MLflow tracking
mlflow.set_experiment("iris-classification")

with mlflow.start_run(run_name="random-forest-v1"):
    # Parameters
    n_estimators = 100
    max_depth = 5

    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Train
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)

    # Save model
    mlflow.sklearn.log_model(model, "model")

    print(f"Accuracy: {accuracy:.3f}")
    print(classification_report(y_test, y_pred))
```

### Basic Regression Pipeline

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load California housing
from datasets import load_dataset
dataset = load_dataset("scikit-learn/california-housing", split="train")
df = dataset.to_pandas()

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

mlflow.set_experiment("housing-regression")

with mlflow.start_run(run_name="gradient-boosting-v1"):
    model = GradientBoostingRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(model, "model")

    print(f"RMSE: {rmse:.3f}, R2: {r2:.3f}")
```

### Hyperparameter Tuning with Tracking

```python
from sklearn.model_selection import GridSearchCV

mlflow.set_experiment("hyperparameter-tuning")

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10, None],
}

with mlflow.start_run(run_name="grid-search"):
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring="accuracy"
    )
    grid_search.fit(X_train, y_train)

    # Log best parameters
    for param, value in grid_search.best_params_.items():
        mlflow.log_param(f"best_{param}", value)

    mlflow.log_metric("best_cv_score", grid_search.best_score_)
    mlflow.log_metric("test_accuracy", grid_search.score(X_test, y_test))

    print(f"Best params: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.3f}")
```

---

## PyTorch / GPU Workflows

### Check GPU Availability

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Simple Neural Network

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Prepare data
X_tensor = torch.FloatTensor(X_train_scaled)
y_tensor = torch.LongTensor(y_train.factorize()[0])

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define model
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

# Train on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNN(4, 32, 3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

mlflow.set_experiment("pytorch-classification")

with mlflow.start_run(run_name="simple-nn"):
    mlflow.log_param("hidden_dim", 32)
    mlflow.log_param("lr", 0.01)

    for epoch in range(100):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            mlflow.log_metric("loss", loss.item(), step=epoch)

    print(f"Final loss: {loss.item():.4f}")
```

---

## Coherence Scoring

The coherence module measures semantic drift in knowledge patterns using the **SC formula**:

```
SC = (Availability x Consistency x Stability)^(1/3)
```

- **Availability:** Can the pattern be retrieved? (embedding cosine similarity to corpus centroid)
- **Consistency:** Does it contradict other knowledge? (NLI entailment/contradiction)
- **Stability:** Has it changed over time? (bi-temporal delta -- stubbed until Graphiti data available)

### Prerequisites

```bash
# Install coherence dependencies
pip install -e ".[coherence,mlops]"

# Ensure Ollama is running with nomic-embed-text
ollama serve &
ollama pull nomic-embed-text

# DeBERTa model downloads automatically on first use (~350MB)
```

### Quick Start: Run with Synthetic Data

The module includes built-in synthetic fixtures for testing -- no external data needed.

```bash
# Run v1-embedding experiment (availability only)
python -m data_systems_toolkit.coherence.cli run-v1

# Run v2-nli experiment (availability + NLI consistency)
python -m data_systems_toolkit.coherence.cli run-v2
```

### Score a Single Pattern

```bash
# Score against a corpus file
python -m data_systems_toolkit.coherence.cli score \
    --text "Data validation prevents downstream quality issues" \
    --corpus data/corpus_patterns.json \
    --method v2-nli
```

Output:
```json
{
  "pattern_id": "ad-hoc",
  "availability": 0.8234,
  "consistency": 0.9112,
  "stability": 1.0,
  "composite_score": 0.9087,
  "method": "v2-nli"
}
```

### Run Experiments with Custom Data

Pattern files are JSON arrays:

```json
[
  {"text": "Schema evolution must be backward compatible", "pattern_id": "p1"},
  {"text": "Data lineage enables root cause analysis", "pattern_id": "p2"}
]
```

```bash
# Run v1 with custom data
python -m data_systems_toolkit.coherence.cli run-v1 \
    --input data/test_patterns.json \
    --corpus data/corpus_patterns.json

# Run v2 with custom data
python -m data_systems_toolkit.coherence.cli run-v2 \
    --input data/test_patterns.json \
    --corpus data/corpus_patterns.json
```

### View Results in MLflow

```bash
mlflow ui --host 0.0.0.0
# Open the MLflow web interface in your browser
```

Experiments are logged under `coherence-v1-embedding` and `coherence-v2-nli` with:
- **Parameters:** model names, corpus size, method
- **Metrics:** avg/min/max availability, consistency, composite score
- **Artifacts:** Full results JSON for each run

### Using the Python API

```python
from data_systems_toolkit.coherence import score_pattern, score_batch, Pattern
from data_systems_toolkit.coherence.fixtures import generate_corpus_patterns

# Create patterns
corpus = generate_corpus_patterns()
pattern = Pattern(text="Validate data at ingestion time", pattern_id="test-1")

# Score with v2-nli (availability + consistency)
from data_systems_toolkit.coherence.availability import embed_texts
corpus_embeddings = embed_texts([p.text for p in corpus])

result = score_pattern(pattern, corpus, corpus_embeddings, method="v2-nli")
print(f"SC = {result.composite_score:.3f}")
print(f"  Availability: {result.availability:.3f}")
print(f"  Consistency:  {result.consistency:.3f}")
print(f"  Stability:    {result.stability:.3f}")
```

### Experiment Methods

| Method | Components | Use Case |
|--------|-----------|----------|
| `v1-embedding` | Availability only | Fast baseline, embedding recall |
| `v2-nli` | Availability + Consistency | Catches contradictions, MVP target |

Stretch experiments (v3-judge, v4-hybrid, v5-multi-judge) are planned for future implementation.

### Interpreting Scores

| Score Range | Interpretation |
|-------------|---------------|
| 0.9 - 1.0 | Highly coherent -- pattern aligns well with corpus |
| 0.7 - 0.9 | Moderate coherence -- some drift or weak alignment |
| 0.5 - 0.7 | Low coherence -- significant divergence |
| < 0.5 | Incoherent -- contradicts or is unrelated to corpus |

Thresholds for "drift detected" will be calibrated empirically from experiment results.

---

## Common Patterns

### Save/Load Processed Data

```python
# Save processed data for reuse
df_processed.to_parquet("data/processed/my_data.parquet")

# Load it back
df = pd.read_parquet("data/processed/my_data.parquet")
```

### Notebook -> Script Migration

When a notebook experiment works, extract to a script:

```python
# scripts/train_model.py
import mlflow
from pathlib import Path

def main():
    mlflow.set_experiment("production-model")

    with mlflow.start_run():
        # Your training code here
        pass

if __name__ == "__main__":
    main()
```

### Load a Logged Model

```python
# From MLflow UI, get the run ID
model = mlflow.sklearn.load_model("runs:/<run_id>/model")
predictions = model.predict(X_new)
```

---

## Troubleshooting

### GPU Not Detected

```bash
# Check nvidia-smi works
nvidia-smi

# Verify PyTorch sees GPU
python -c "import torch; print(torch.cuda.is_available())"
```

If False, rebuild container: `Ctrl+Shift+P` -> "Dev Containers: Rebuild Container"

### Out of GPU Memory

```python
# Clear GPU cache
torch.cuda.empty_cache()

# Reduce batch size
loader = DataLoader(dataset, batch_size=16)  # smaller batch

# Use mixed precision
from torch.cuda.amp import autocast
with autocast():
    outputs = model(inputs)
```

### MLflow UI Not Loading

```bash
# Make sure it's running
mlflow ui --host 0.0.0.0

# Check port forwarding in VSCode (Ports tab)
```

---

## Next Steps

1. Run `notebooks/00-environment-test.ipynb` to verify setup
2. Create a new notebook in `notebooks/` for your experiment
3. Load a dataset from HuggingFace
4. Train a model with MLflow tracking
5. Compare runs in MLflow UI

For stack simulation and lineage tracking (Phase 2), see the project specs.
