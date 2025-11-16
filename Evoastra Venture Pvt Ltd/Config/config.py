"""
Configuration file for Image Caption Generator
"""
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, "models")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
MODEL_WEIGHTS_PATH = os.path.join(MODEL_DIR, "image_captioning_model_weights.weights.h5")

# Model parameters (must match training parameters)
# IMPORTANT: This must match the max_length used during training
# From training notebook: max_length = df['clean_caption_length'].max() + 2 = 34
# But model was likely trained with 36 as buffer. Adjust if needed.
MAX_LENGTH = 36
VOCAB_SIZE = None  # Will be determined from tokenizer
EMBEDDING_DIM = 256
LSTM_UNITS = 256
DROPOUT_RATE = 0.4

# Image preprocessing
IMAGE_SIZE = (299, 299)  # InceptionV3 input size

# Feature extractor
FEATURE_EXTRACTOR_MODEL = "InceptionV3"
FEATURE_VECTOR_SIZE = 2048

