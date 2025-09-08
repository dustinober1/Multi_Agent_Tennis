# 🚀 Deployment Guide

This guide covers deploying the Multi-Agent Tennis project for production use.

## 📖 Documentation Deployment

### GitHub Pages
1. Build documentation: `./build_docs.sh`
2. Deploy to GitHub Pages via repository settings
3. Documentation available at: `https://username.github.io/Multi_Agent_Tennis`

### Self-hosted
```bash
cd docs/_build/html
python -m http.server 8000
```

## 🌐 Web Demo Deployment

### Local Development
```bash
streamlit run src/web_demo.py --server.port 8501
```

### Streamlit Cloud
1. Push to GitHub repository
2. Connect to Streamlit Cloud
3. Deploy from main branch with entry point: `src/web_demo.py`

### Docker
```dockerfile
FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "src/web_demo.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
```

## Production Features Added ✅

- **Professional Documentation** with Sphinx
- **Interactive Web Demo** with Streamlit  
- **Enhanced API Documentation** with improved docstrings
- **Deployment Scripts** for easy setup
- **Configuration Management** integration
- **Real-time Visualizations** and analytics
