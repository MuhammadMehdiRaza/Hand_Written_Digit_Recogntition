# 🚀 Installation Guide

## Python Version Requirements

**IMPORTANT**: TensorFlow currently supports Python 3.9 - 3.11. You have Python 3.14.2 installed, which is not yet supported by TensorFlow.

## Option 1: Install Python 3.11 (Recommended)

### Windows:

1. Download Python 3.11 from [python.org](https://www.python.org/downloads/)
2. During installation, check "Add Python to PATH"
3. Open a new terminal and verify: `py -3.11 --version`
4. Create a virtual environment:
   ```bash
   py -3.11 -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

### After Installing Python 3.11:

```bash
# Navigate to project
cd Hand_Written_Digit_Recogntition

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Train the model
python train_model.py

# Run the app
streamlit run app.py
```

## Option 2: Use Conda (Alternative)

If you have Anaconda or Miniconda installed:

```bash
# Create conda environment with Python 3.11
conda create -n digit_recognition python=3.11
conda activate digit_recognition

# Install TensorFlow
conda install -c conda-forge tensorflow

# Install other packages
pip install streamlit streamlit-drawable-canvas opencv-python pillow matplotlib numpy

# Train and run
python train_model.py
streamlit run app.py
```

## Option 3: Use Google Colab (No Local Installation)

If you can't install Python 3.11 locally, use Google Colab:

1. Upload `train_model.py` to Colab
2. Run training in Colab
3. Download the trained model
4. Place it in the `models/` folder locally

## Troubleshooting

### TensorFlow Installation Issues

**For Windows:**
```bash
pip install tensorflow-cpu  # CPU-only version, faster to install
```

**For Mac (Intel):**
```bash
pip install tensorflow
```

**For Mac (M1/M2/M3 - Apple Silicon):**
```bash
pip install tensorflow-macos
pip install tensorflow-metal  # For GPU acceleration
```

**For Linux:**
```bash
pip install tensorflow
```

### Streamlit Canvas Issues

If the canvas doesn't work:
```bash
pip install --upgrade streamlit streamlit-drawable-canvas
streamlit cache clear
```

### OpenCV Issues

```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

## Verify Installation

```bash
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
python -c "import streamlit as st; print(f'Streamlit version: {st.__version__}')"
```

## Quick Start After Installation

```bash
# 1. Train the model (5-10 minutes)
python train_model.py

# 2. Run the Streamlit app
streamlit run app.py

# 3. Open browser at http://localhost:8501
```

## Pre-trained Model

If you don't want to train the model yourself, you can use a pre-trained model:

1. Download from [Google Drive](#) (link to be added)
2. Place in `models/digit_recognition_model.keras`
3. Run `streamlit run app.py`

---

For more help, open an issue on GitHub!
