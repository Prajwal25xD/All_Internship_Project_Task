"""
Utility functions for loading models and tokenizer
"""
import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.applications.inception_v3 import InceptionV3
from config.config import (
    TOKENIZER_PATH, MODEL_WEIGHTS_PATH, MAX_LENGTH,
    EMBEDDING_DIM, LSTM_UNITS, DROPOUT_RATE, FEATURE_VECTOR_SIZE
)


def load_tokenizer():
    """Load the tokenizer from pickle file."""
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}")
    
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    
    return tokenizer


def build_model(vocab_size, max_length):
    """
    Build the image captioning model architecture.
    This must match the architecture used during training.
    """
    # Image feature input
    input_img_features = Input(shape=(FEATURE_VECTOR_SIZE,), name="image_input")
    feat_model = Dropout(DROPOUT_RATE)(input_img_features)
    feat_model = Dense(EMBEDDING_DIM, activation='relu')(feat_model)

    # Text sequence input
    input_text = Input(shape=(max_length,), name="text_input")
    text_model = Embedding(vocab_size, EMBEDDING_DIM, mask_zero=True)(input_text)
    text_model = Dropout(DROPOUT_RATE)(text_model)
    text_model = LSTM(LSTM_UNITS)(text_model)

    # Combine the two inputs
    decoder = add([feat_model, text_model])
    decoder = Dense(EMBEDDING_DIM, activation='relu')(decoder)
    output = Dense(vocab_size, activation='softmax')(decoder)

    # Create the final model
    model = Model(inputs=[input_img_features, input_text], outputs=output)
    
    return model


def load_caption_model():
    """Load the trained caption model with weights."""
    # Load tokenizer
    tokenizer = load_tokenizer()
    
    # Get vocabulary size
    vocab_size = len(tokenizer.word_index) + 1
    
    # Build model architecture
    model = build_model(vocab_size, MAX_LENGTH)
    
    # Compile model (required for inference, even if not training)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    # Load weights
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        raise FileNotFoundError(f"Model weights not found at {MODEL_WEIGHTS_PATH}")
    
    model.load_weights(MODEL_WEIGHTS_PATH)
    
    # Create reverse index for faster word lookup
    # Keras Tokenizer has index_word attribute, but if not available, create it
    if not hasattr(tokenizer, 'index_word') or tokenizer.index_word is None:
        tokenizer.index_word = {index: word for word, index in tokenizer.word_index.items()}
    
    return model, tokenizer


def load_feature_extractor():
    """Load InceptionV3 model for feature extraction."""
    base_model = InceptionV3(weights='imagenet')
    feature_extractor = Model(base_model.input, base_model.layers[-2].output)
    return feature_extractor

