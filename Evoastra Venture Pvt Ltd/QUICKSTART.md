# Quick Start Guide

Get the Image Caption Generator up and running in 5 minutes!

## ⚡ Quick Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Model Files

Check that these files exist in `models/`:
- ✅ `tokenizer.pkl`
- ✅ `image_captioning_model_weights.weights.h5`

If missing, copy them from `notebooks/`:
```bash
# Windows PowerShell
Copy-Item notebooks\tokenizer.pkl models\
Copy-Item notebooks\image_captioning_model_weights.weights.h5 models\

# Linux/Mac
cp notebooks/tokenizer.pkl models/
cp notebooks/image_captioning_model_weights.weights.h5 models/
```

### 3. Run the App

```bash
streamlit run app.py
```

That's it! The app should open in your browser at `http://localhost:8501`

## 🎯 Using the App

1. **Upload an Image**: Click "Choose an image..." or drag & drop
2. **Click "Generate Caption"**: Wait a few seconds
3. **View Result**: See the generated caption below the image

## 🐛 Quick Troubleshooting

**App won't start?**
- Check Python version: `python --version` (need 3.8+)
- Install dependencies: `pip install -r requirements.txt`

**Model files not found?**
- Ensure files are in `models/` directory
- Check file names match exactly

**Port already in use?**
```bash
streamlit run app.py --server.port=8502
```

## 📚 Next Steps

- Read [README.md](README.md) for detailed documentation
- Check [DEPLOYMENT.md](DEPLOYMENT.md) for deployment options
- Explore the code in `utils/` and `config/` directories

---

**Need help?** Check the full [README.md](README.md) or open an issue.

