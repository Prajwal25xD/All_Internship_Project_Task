import streamlit as st
import numpy as np
import pickle
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- 1. Load Models and Tokenizer (Cached) ---
# We use @st.cache_resource to load these heavy objects only once
@st.cache_resource
def load_all():
    """Loads the tokenizer, model, and feature extractor."""
    
    # --- Load Tokenizer ---
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    # --- Get Vocab Size and Max Length ---
    # WARNING: Hardcoding max_length. 
    # This MUST be the same value used in training (e.g., 36)
    # A better way is to save this value in a .json or with the tokenizer.
    vocab_size = len(tokenizer.word_index) + 1
    max_length = 36 # <-- IMPORTANT: Change this if your max_length was different
    
    # --- Define Model Architecture (MUST match training) ---
    input_img_features = Input(shape=(2048,), name="image_input")
    feat_model = Dropout(0.4)(input_img_features)
    feat_model = Dense(256, activation='relu')(feat_model)

    input_text = Input(shape=(max_length,), name="text_input")
    text_model = Embedding(vocab_size, 256, mask_zero=True)(input_text)
    text_model = Dropout(0.4)(text_model)
    text_model = LSTM(256)(text_model)

    decoder = add([feat_model, text_model])
    decoder = Dense(256, activation='relu')(decoder)
    output = Dense(vocab_size, activation='softmax')(decoder)
    
    model = Model(inputs=[input_img_features, input_text], outputs=output)
    
    # --- Load Trained Weights ---
    model.load_weights("image_captioning_model_weights.weights.h5")
    
    # --- Load InceptionV3 Feature Extractor ---
    base_model = In_ceptionV3(weights='imagenet')
    feature_extractor = Model(base_model.input, base_model.layers[-2].output)
    
    print("--- All models loaded successfully! ---")
    return model, feature_extractor, tokenizer, max_length

# --- 2. Define Helper Functions ---

def preprocess_pil_image(image_pil):
    """Preprocesses a PIL Image for InceptionV3."""
    # Resize to 299x299
    image = image_pil.resize((299, 299))
    
    # Convert to 3-channel RGB if it's grayscale
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_array)
    return img_preprocessed

def extract_features_from_pil(image_pil, feature_extractor_model):
    """Extracts features from a PIL image."""
    img_preprocessed = preprocess_pil_image(image_pil)
    features = feature_extractor_model.predict(img_preprocessed, verbose=0)
    return features

def word_for_id(integer, tokenizer):
    """Converts a token ID back to its word."""
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_caption(model, tokenizer, image_features, max_length):
    """Generates a caption for an image using greedy search."""
    in_text = '<start>'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        
        yhat = model.predict([image_features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        
        if word is None:
            break
        in_text += ' ' + word
        if word == '<end>':
            break
            
    final_caption = in_text.replace('<start>', '').replace('<end>', '').strip()
    return final_caption

# --- 3. Streamlit UI ---

# Load all the models and assets
model, feature_extractor, tokenizer, max_length = load_all()

st.set_page_config(page_title="Image Caption Generator", layout="centered")
st.title("🖼️ Image Caption Generator")
st.write("Upload an image and this AI model will generate a descriptive caption for it, based on the Flickr8k dataset.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # 2. Generate and display the caption
    with st.spinner('AI is thinking... 🧠'):
        # Extract features
        image_features = extract_features_from_pil(image, feature_extractor)
        
        # Generate caption
        caption = generate_caption(model, tokenizer, image_features, max_length)
        
        st.subheader("Generated Caption:")
        st.write(f"**{caption.capitalize()}**")