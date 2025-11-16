# Deployment Guide

This guide will help you deploy the Image Caption Generator Streamlit app.

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- Model files (tokenizer.pkl and model weights)

## 🚀 Local Deployment

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Verify Model Files

Ensure the following files exist in the `models/` directory:
- `tokenizer.pkl`
- `image_captioning_model_weights.weights.h5`

If they're missing, run:
```bash
python setup.py
```

### Step 3: Run the Application

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

## ☁️ Cloud Deployment

### Option 1: Streamlit Cloud (Recommended)

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Important Notes for Streamlit Cloud**
   - Model files must be in the repository (or use external storage)
   - File size limit: 1GB per file
   - Consider using Git LFS for large model files

### Option 2: Heroku

1. **Create Procfile**
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Create runtime.txt**
   ```
   python-3.10.0
   ```

3. **Deploy**
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### Option 3: Docker

1. **Create Dockerfile**
   ```dockerfile
   FROM python:3.10-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   COPY . .
   
   EXPOSE 8501
   
   CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Build and Run**
   ```bash
   docker build -t caption-generator .
   docker run -p 8501:8501 caption-generator
   ```

### Option 4: AWS EC2 / Google Cloud / Azure

1. **Set up a virtual machine**
2. **Install Python and dependencies**
3. **Clone your repository**
4. **Run the Streamlit app**
5. **Configure firewall to allow port 8501**

## 🔧 Configuration

### Environment Variables

You can set environment variables for configuration:

```bash
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Custom Port

```bash
streamlit run app.py --server.port=8080
```

## 📊 Performance Optimization

### For Production:

1. **Use GPU** (if available)
   - TensorFlow will automatically use GPU if available
   - Significantly faster inference

2. **Enable Caching**
   - Already implemented with `@st.cache_resource`
   - Models are loaded once and cached

3. **Reduce Model Size**
   - Consider model quantization
   - Use TensorFlow Lite for mobile deployment

4. **CDN for Static Assets**
   - Serve images and static files via CDN

## 🐛 Troubleshooting

### Port Already in Use

```bash
# Find process using port 8501
lsof -i :8501  # Linux/Mac
netstat -ano | findstr :8501  # Windows

# Kill the process or use a different port
streamlit run app.py --server.port=8502
```

### Model Loading Errors

- Verify model files exist in `models/` directory
- Check file paths in `config/config.py`
- Ensure model architecture matches training

### Memory Issues

- Close other applications
- Use a machine with more RAM
- Consider using a smaller batch size

## 📝 Notes

- First run may be slower due to model loading
- InceptionV3 weights will be downloaded on first run (~100MB)
- Ensure sufficient disk space for model files

## 🔒 Security Considerations

- Don't commit API keys or secrets
- Use environment variables for sensitive data
- Enable authentication for production deployments
- Use HTTPS in production

---

For more information, see the [README.md](README.md) file.

