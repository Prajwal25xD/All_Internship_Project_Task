# Project Structure

Complete overview of the Image Caption Generator project structure.

## 📁 Directory Structure

```
caption_generator/
│
├── app.py                          # Main Streamlit application (ENTRY POINT)
├── requirements.txt                # Python dependencies
├── setup.py                        # Setup script for project initialization
├── README.md                       # Main documentation
├── QUICKSTART.md                   # Quick start guide
├── DEPLOYMENT.md                   # Deployment instructions
├── PROJECT_STRUCTURE.md            # This file
├── .gitignore                      # Git ignore rules
│
├── .streamlit/                     # Streamlit configuration
│   └── config.toml                 # Streamlit app settings
│
├── config/                         # Configuration module
│   ├── __init__.py
│   └── config.py                   # Model parameters and paths
│
├── utils/                          # Utility modules
│   ├── __init__.py
│   ├── model_loader.py             # Model loading functions
│   ├── image_processing.py         # Image preprocessing
│   └── caption_generator.py        # Caption generation logic
│
├── models/                         # Model files (NOT in git)
│   ├── tokenizer.pkl               # Trained tokenizer
│   └── image_captioning_model_weights.weights.h5  # Model weights
│
├── src/                            # Source code (reserved for future use)
│   └── __init__.py
│
├── datasets/                       # Dataset files
│   ├── captions.txt                # Original captions
│   └── clean_df(EDA)_1.csv        # Cleaned dataset
│
├── Images/                         # Image dataset (NOT in git)
│   └── [8091 .jpg files]
│
└── notebooks/                      # Jupyter notebooks for training
    ├── app.py                      # Original app (deprecated)
    ├── image_preprocessed.ipynb    # Image preprocessing notebook
    ├── captions_preprocessed.ipynb # Caption preprocessing notebook
    ├── model_training.ipynb        # Model training notebook
    ├── tokenizer.pkl               # Original tokenizer (backup)
    ├── image_captioning_model_weights.weights.h5  # Original weights (backup)
    ├── image_features.pkl          # Pre-extracted features
    └── requirements.txt            # Original requirements (backup)
```

## 🔑 Key Files

### Entry Point
- **`app.py`**: Main Streamlit application. Run with `streamlit run app.py`

### Configuration
- **`config/config.py`**: All model parameters, paths, and settings
- **`.streamlit/config.toml`**: Streamlit UI configuration

### Core Modules
- **`utils/model_loader.py`**: Loads models and tokenizer
- **`utils/image_processing.py`**: Image preprocessing functions
- **`utils/caption_generator.py`**: Caption generation logic

### Model Files (Required)
- **`models/tokenizer.pkl`**: Text tokenizer (must exist)
- **`models/image_captioning_model_weights.weights.h5`**: Model weights (must exist)

## 📦 Module Dependencies

```
app.py
├── config.config
│   └── (paths and parameters)
├── utils.model_loader
│   ├── config.config
│   └── (loads models)
├── utils.image_processing
│   ├── config.config
│   └── (preprocesses images)
└── utils.caption_generator
    ├── config.config
    └── (generates captions)
```

## 🚀 Deployment Files

- **`requirements.txt`**: All Python dependencies
- **`setup.py`**: Automated setup script
- **`DEPLOYMENT.md`**: Deployment instructions
- **`.gitignore`**: Excludes large files and sensitive data

## 📝 Documentation Files

- **`README.md`**: Complete project documentation
- **`QUICKSTART.md`**: Quick start guide
- **`DEPLOYMENT.md`**: Deployment guide
- **`PROJECT_STRUCTURE.md`**: This file

## 🔒 Files Excluded from Git

The following are in `.gitignore`:
- `models/*.pkl`, `models/*.h5` (large model files)
- `Images/` (image dataset)
- `*.csv` (dataset files)
- `__pycache__/`, `*.pyc` (Python cache)
- `.venv/`, `venv/` (virtual environments)

## ✅ Verification Checklist

Before running the app, verify:

- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `models/tokenizer.pkl` exists
- [ ] `models/image_captioning_model_weights.weights.h5` exists
- [ ] All directories created (`config/`, `utils/`, `models/`)

## 🎯 Running the App

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify model files exist
ls models/

# 3. Run the app
streamlit run app.py
```

## 📊 File Sizes (Approximate)

- Model weights: ~50-200 MB
- Tokenizer: ~1-5 MB
- Image dataset: ~1-2 GB (not required for app)
- Code files: < 1 MB total

---

For more information, see [README.md](README.md) or [QUICKSTART.md](QUICKSTART.md).

