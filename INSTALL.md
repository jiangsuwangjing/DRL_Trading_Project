# Installation Guide

## Quick Fix for Python 3.13 Issue

If you're encountering PyTorch installation issues with Python 3.13, follow these steps:

### Option 1: Use Python 3.12 (Recommended)

```bash
# Install Python 3.12 using uv
uv python install 3.12

# Create a virtual environment with Python 3.12
uv venv --python 3.12

# Activate the environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows

# Install dependencies
uv pip install -r requirements.txt
```

### Option 2: Use Conda

```bash
# Create conda environment with Python 3.12
conda create -n stock_rl python=3.12
conda activate stock_rl

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Manual Torch Installation (Advanced)

If you absolutely need Python 3.13, you can try installing PyTorch from source or nightly builds:

```bash
# Install PyTorch nightly (may work with Python 3.13)
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

# Then install other dependencies
uv pip install -r requirements.txt --no-deps
uv pip install numpy pandas gymnasium stable-baselines3 yfinance matplotlib seaborn scipy scikit-learn pandas-ta tqdm joblib pyyaml
```

**Note**: Option 3 is experimental and not guaranteed to work. Option 1 (Python 3.12) is strongly recommended.

## Verify Installation

After installation, run:

```bash
python test_setup.py
```

This will verify that all dependencies are correctly installed.

