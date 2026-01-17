# Project Seminar Scientific Working

This repository contains the code to improve object detection performance using GAN-generated CT images.

## Package installation with uv

[`uv`](https://docs.astral.sh/uv/) is an extremely fast Python package installer and resolver, written in Rust. It's recommended for its speed and reliability.

**Step 1: Install `uv`**

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

**Step 2: Create and activate virtual environment**

```bash
# Create a Python 3.12 virtual environment
uv venv --python 3.12

# Activate the environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows
```

**Step 3: Install dependencies**

```bash
# Install all dependencies from requirements.txt
uv pip install -r requirements.txt
```

**Step 4: Install the package in editable mode**

```bash
cd code
uv pip install -e .
```

## Project setup
```bash
# Switch into the code folder
cd code

# Run the download script to get the dataset
# Please note that not all Operating Systems support bash - in that case, please download the dataset another way
bash download.sh

# Fork the CycleGAN repository
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git

# Run the project
project
```