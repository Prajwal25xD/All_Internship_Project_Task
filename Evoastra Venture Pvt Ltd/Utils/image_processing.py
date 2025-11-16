"""
Utility functions for image preprocessing and feature extraction
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input
from PIL import Image
from config.config import IMAGE_SIZE


def preprocess_pil_image(image_pil):
    """
    Preprocesses a PIL Image for InceptionV3.
    
    Args:
        image_pil: PIL Image object
        
    Returns:
        Preprocessed image array ready for InceptionV3
    """
    # Resize to 299x299
    image = image_pil.resize(IMAGE_SIZE)
    
    # Convert to 3-channel RGB if it's grayscale
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Convert to array and add batch dimension
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess for InceptionV3
    img_preprocessed = preprocess_input(img_array)
    
    return img_preprocessed


def extract_features_from_pil(image_pil, feature_extractor_model):
    """
    Extracts features from a PIL image using the feature extractor model.
    
    Args:
        image_pil: PIL Image object
        feature_extractor_model: Keras model for feature extraction
        
    Returns:
        Feature vector of shape (1, 2048)
    """
    img_preprocessed = preprocess_pil_image(image_pil)
    features = feature_extractor_model.predict(img_preprocessed, verbose=0)
    return features

