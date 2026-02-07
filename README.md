# End-to-End Speech Enhancement Research Framework

This repository implements a fully automated pipeline for developing, training, evaluating, and tracking Speech Enhancement (SE) models. It is designed to streamline the iteration process for deep learning algorithms.

## Workflow Overview

The core of the workflow is `manager.py`, which automates the following steps:
1.  **Training**: Trains the model using specified configurations.
2.  **Inference**: Automatically generates enhanced audio using the best checkpoint.
3.  **Evaluation**: Calculates objective metrics (PESQ, STOI, SISNR, DNSMOS, etc.).
4.  **Logging**: Records experiment details and results into a global `data_log/exp_table.md`.

## 🚀 Quick Start

### 1. Create a New Model

To add a new model algorithm, create a Python file in the `models_new/` directory.

**Example**: `models_new/my_super_model.py`

```python
import torch
import torch.nn as nn

class SuperNet(nn.Module):
    def __init__(self, n_fft=512, hop_len=256, win_len=512):
        super().__init__()
        # Your model initialization
        # Note: The constructor arguments must match 'network_config' in configs/cfg_train_auto.yaml
    
    def forward(self, x):
        # x shape: [B, F, T, 2] usually, depending on configuration
        return x
```

### 2. Start an Experiment

Use `manager.py` to start the pipeline. You don't need to modify configuration files manually; you can override settings via command line arguments.

**Command Syntax**:
```bash
python manager.py start --gpu <GPU_ID> --name "<Experiment_Name>" [overrides]
```

**Example**:
To train the `SuperNet` class defined in `models_new/my_super_model.py`:

```bash
python manager.py start --gpu 0 --name "SuperNet_v1" \
    model.path=models_new.my_super_model \
    model.classname=SuperNet
```

**Key Parameters**:
*   `--gpu`: GPU index to use (e.g., `0`, `1`).
*   `--name`: A human-readable name for the experiment (will appear in logs and charts).
*   `model.path`: Python import path to your model file (e.g. `models_new.filename`).
*   `model.classname`: Name of the class to instantiate.

### 3. Monitor Progress

**List Experiments**:
View all experiments and their status:
```bash
python manager.py list
```

**TensorBoard**:
Visualize loss curves and audio samples. You can filter by experiment ID or name keywords.
```bash
# Show specific known experiments
python manager.py board exp_gtcrn_2026-02-04 exp_supernet_v1

# Show experiments matching a keyword
python manager.py board SuperNet
```

### 4. Check Results

Once the pipeline finishes (`Training` -> `Inference` -> `Evaluation`), the results are automatically saved:

1.  **Global Table**: Check `data_log/exp_table.md`. This file contains a markdown table comparing all experiments with metrics like SDR, SISNR, PESQ, STOI, etc.
2.  **Experiment Folder**: Go to `experiments/exp_<model>_<timestamp>/`.
    *   `checkpoints/`: Saved models.
    *   `logs/`: TensorBoard logs.
    *   `config.yaml`: Configuration used.
    *   `metadata.json`: Experiment metadata and metrics.
    *   `best_model_xxx/enhanced_dns/`: Generated audio files.
    *   `best_model_xxx/enhanced_dns/scoring_*/RESULTS.txt`: Detailed metric logs.

## 📂 Project Structure

*   `manager.py`: Main entry point for the automation pipeline.
*   `models_new/`: Place your new model architectures here.
*   `models/`: Legacy models.
*   `configs/`: Configuration files (YAML).
    *   `cfg_train_auto.yaml`: Base configuration for training.
*   `train_auto.py`: Underlying training script (called by manager).
*   `infer_auto.py`: Underlying inference script (called by manager).
*   `evaluate.py`: Evaluation script (called by manager).
*   `experiments/`: Directory where all experiment run data is stored.
*   `data_log/`: Stores the global results table (`exp_table.md`).

## Configuration System

The project uses `OmegaConf`. The base configuration is `configs/cfg_train_auto.yaml`.

Common overrides you might use:
*   `optimizer.lr=0.0005`: Change learning rate.
*   `train_dataloader.batch_size=8`: Change batch size.
*   `loss.lamda_mag=50`: Adjust loss weights.

Example with multiple overrides:
```bash
python manager.py start --gpu 1 --name test_exp \
    model.path=models_new.my_super_model \
    model.classname=SuperNet \
    optimizer.lr=5e-4 \
    train_dataloader.batch_size=32
```
