"""
Setup script for Image Caption Generator
"""
import os
import shutil
from pathlib import Path

def setup_project():
    """Setup the project structure and move model files."""
    base_dir = Path(__file__).parent
    
    # Create models directory if it doesn't exist
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Move model files from notebooks to models directory
    notebooks_dir = base_dir / "notebooks"
    
    files_to_move = [
        ("tokenizer.pkl", "tokenizer.pkl"),
        ("image_captioning_model_weights.weights.h5", "image_captioning_model_weights.weights.h5")
    ]
    
    for src_name, dst_name in files_to_move:
        src_path = notebooks_dir / src_name
        dst_path = models_dir / dst_name
        
        if src_path.exists() and not dst_path.exists():
            print(f"Moving {src_name} to models/ directory...")
            shutil.copy2(src_path, dst_path)
            print(f"✓ {src_name} moved successfully")
        elif dst_path.exists():
            print(f"✓ {dst_name} already exists in models/ directory")
        else:
            print(f"⚠ Warning: {src_name} not found in notebooks/ directory")
    
    print("\n✓ Project setup complete!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run the app: streamlit run app.py")

if __name__ == "__main__":
    setup_project()

