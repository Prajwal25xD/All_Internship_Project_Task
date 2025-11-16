# Image Caption Generator

A deep learning-based image captioning application that generates descriptive captions for images using a CNN-LSTM architecture. The model is trained on the Flickr8k dataset and deployed as a Streamlit web application.

## 🎯 Features

- **Automatic Image Captioning**: Upload any image and get an AI-generated caption
- **Deep Learning Model**: Uses InceptionV3 for feature extraction and LSTM for caption generation
- **User-Friendly Interface**: Clean and intuitive Streamlit web interface
- **Real-Time Processing**: Fast caption generation with progress indicators

## 📋 Project Structure

```
caption_generator/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .gitignore           # Git ignore file
├── config/              # Configuration files
│   └── config.py        # Model and path configurations
├── utils/               # Utility modules
│   ├── model_loader.py  # Model loading functions
│   ├── image_processing.py  # Image preprocessing
│   └── caption_generator.py # Caption generation logic
├── models/              # Model files (not in repo)
│   ├── tokenizer.pkl
│   └── image_captioning_model_weights.weights.h5
├── datasets/            # Dataset files
│   ├── captions.txt
│   └── clean_df(EDA)_1.csv
├── Images/              # Image dataset (not in repo)
└── notebooks/           # Jupyter notebooks for training
    ├── image_preprocessed.ipynb
    ├── captions_preprocessed.ipynb
    └── model_training.ipynb
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd caption_generator
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data (if needed)

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## 📦 Model Files Setup

Before running the application, ensure you have the following model files in the `models/` directory:

1. **tokenizer.pkl**: The trained tokenizer for text processing
2. **image_captioning_model_weights.weights.h5**: The trained model weights

If you don't have these files, you can:
- Train the model using the notebooks in the `notebooks/` directory
- Download pre-trained weights (if available)

## 🎮 Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Application

1. **Upload an Image**: Click on "Choose an image..." or drag and drop an image file
2. **Generate Caption**: Click the "Generate Caption" button
3. **View Results**: The generated caption will appear below the image

### Supported Image Formats

- JPG/JPEG
- PNG
- BMP
- WebP

## 🏗️ Model Architecture

The image captioning model uses a two-part architecture:

1. **Feature Extractor (CNN)**: 
   - InceptionV3 pre-trained on ImageNet
   - Extracts 2048-dimensional feature vectors from images

2. **Caption Generator (LSTM)**:
   - Embedding layer for word representations
   - LSTM layer for sequence modeling
   - Dense layers for output prediction

### Model Parameters

- **Max Caption Length**: 36 words
- **Embedding Dimension**: 256
- **LSTM Units**: 256
- **Dropout Rate**: 0.4
- **Vocabulary Size**: ~8764 words

## 📊 Training

The model was trained on the Flickr8k dataset:
- **Training Images**: 6,472
- **Validation Images**: 1,619
- **Total Images**: 8,091
- **Captions per Image**: 5
- **Total Captions**: ~40,455

To retrain the model, refer to the notebooks in the `notebooks/` directory:
1. `image_preprocessed.ipynb`: Image feature extraction
2. `captions_preprocessed.ipynb`: Text preprocessing
3. `model_training.ipynb`: Model training

## 🔧 Configuration

Model parameters can be adjusted in `config/config.py`:

```python
MAX_LENGTH = 36          # Maximum caption length
EMBEDDING_DIM = 256      # Embedding dimension
LSTM_UNITS = 256         # LSTM units
DROPOUT_RATE = 0.4       # Dropout rate
```

## 🐛 Troubleshooting

### Common Issues

1. **Model files not found**
   - Ensure `tokenizer.pkl` and model weights are in the `models/` directory
   - Check file paths in `config/config.py`

2. **Memory errors**
   - Reduce batch size or use a machine with more RAM
   - Close other applications to free up memory

3. **Slow caption generation**
   - First run may be slower due to model loading
   - Consider using GPU acceleration for faster inference

4. **Import errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

## 📝 License

This project is for educational purposes. Please ensure you have proper licenses for:
- Flickr8k dataset
- TensorFlow/Keras
- Streamlit

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on the repository.

## 🙏 Acknowledgments

- Flickr8k dataset for training data
- TensorFlow/Keras for deep learning framework
- Streamlit for web application framework
- InceptionV3 pre-trained model from Google

---

**Note**: This is a research/educational project. For production use, consider additional optimizations and error handling.

