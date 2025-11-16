"""
Streamlit App for Image Caption Generator
"""
import streamlit as st
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
from utils.model_loader import load_caption_model, load_feature_extractor
from utils.image_processing import extract_features_from_pil
from utils.caption_generator import generate_caption
from config.config import MAX_LENGTH


# --- Page Configuration ---
st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Load Models and Tokenizer (Cached) ---
# We use @st.cache_resource to load these heavy objects only once
@st.cache_resource
def load_all():
    """Loads the tokenizer, model, and feature extractor."""
    try:
        # Load caption model and tokenizer
        model, tokenizer = load_caption_model()
        
        # Load feature extractor
        feature_extractor = load_feature_extractor()
        
        st.success("✅ All models loaded successfully!")
        return model, feature_extractor, tokenizer, MAX_LENGTH
    except FileNotFoundError as e:
        st.error(f"❌ Error loading models: {e}")
        st.error("Please ensure all model files are in the 'models/' directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()


# --- Main App ---
def main():
    """Main Streamlit application."""
    
    # Sidebar
    with st.sidebar:
        st.title("🖼️ Image Caption Generator")
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This app generates captions for images using a deep learning model 
        trained on the Flickr8k dataset.
        
        **How it works:**
        1. Upload an image
        2. The model extracts visual features
        3. A caption is generated using an LSTM-based decoder
        """)
        st.markdown("---")
        st.markdown("### Model Info")
        st.markdown(f"**Max Caption Length:** {MAX_LENGTH}")
        st.markdown("**Feature Extractor:** InceptionV3")
        st.markdown("**Architecture:** CNN + LSTM")
    
    # Main content
    st.title("🖼️ Image Caption Generator")
    st.markdown("Upload an image and this AI model will generate a descriptive caption for it.")
    st.markdown("---")
    
    # Load models (cached)
    try:
        model, feature_extractor, tokenizer, max_length = load_all()
    except:
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Upload an image file to generate a caption"
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Generation options
            col1, col2 = st.columns([3, 1])
            with col1:
                use_beam_search = st.checkbox("Use Beam Search (better quality, slower)", value=False)
            with col2:
                beam_size = st.number_input("Beam Size", min_value=1, max_value=5, value=3, step=1) if use_beam_search else 1
            
            # Generate caption button
            if st.button("Generate Caption", type="primary", use_container_width=True):
                # Generate and display the caption
                with st.spinner('AI is thinking... 🧠 Generating caption...'):
                    try:
                        # Extract features
                        image_features = extract_features_from_pil(image, feature_extractor)
                        
                        # Generate caption with selected method
                        if use_beam_search and beam_size > 1:
                            from utils.caption_generator import generate_caption_beam_search
                            caption = generate_caption_beam_search(model, tokenizer, image_features, max_length, beam_size)
                        else:
                            caption = generate_caption(model, tokenizer, image_features, max_length, beam_size=1)
                        
                        # Display results
                        st.markdown("---")
                        st.subheader("📝 Generated Caption:")
                        st.markdown(f"### {caption.capitalize()}")
                        
                        # Show caption info
                        if caption:
                            st.info(f"**Length:** {len(caption.split())} words | **Method:** {'Beam Search' if use_beam_search else 'Greedy Search'}")
                        
                        # Success message
                        st.success("Caption generated successfully! ✨")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating caption: {e}")
                        st.exception(e)
        
        except Exception as e:
            st.error(f"❌ Error loading image: {e}")
            st.exception(e)
    
    else:
        # Show example or instructions
        st.info("👆 Please upload an image to get started!")
        
        # Example section
        with st.expander("ℹ️ How to use"):
            st.markdown("""
            1. **Click** on the "Choose an image..." area or drag and drop an image
            2. **Select** an image file (JPG, PNG, etc.)
            3. **Click** the "Generate Caption" button
            4. **Wait** for the AI to generate a caption (usually takes a few seconds)
            5. **View** the generated caption below the image
            """)


if __name__ == "__main__":
    main()

