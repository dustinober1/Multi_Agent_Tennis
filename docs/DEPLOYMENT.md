# 🚀 Deployment Guide

This guide covers deploying the Multi-Agent Tennis project for production use.

## 📖 Documentation Deployment

### GitHub Pages

1. **Build documentation**:
```bash
./build_docs.sh
```

2. **Deploy to GitHub Pages**:
```bash
# Create gh-pages branch
git checkout --orphan gh-pages
git rm -rf .
cp -r docs/_build/html/* .
git add .
git commit -m "Deploy documentation"
git push origin gh-pages
```

3. **Enable GitHub Pages** in repository settings

### Self-hosted Documentation

Use the built-in Python server:
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

1. **Push to GitHub repository**
2. **Connect to Streamlit Cloud**
3. **Deploy from main branch**
4. **Set entry point**: `src/web_demo.py`

### Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "src/web_demo.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
```

Build and run:
```bash
docker build -t tennis-demo .
docker run -p 8501:8501 tennis-demo
```

### Heroku Deployment

1. **Create `runtime.txt`**:
```
python-3.8.16
```

2. **Create `Procfile`**:
```
web: streamlit run src/web_demo.py --server.port $PORT --server.address 0.0.0.0
```

3. **Deploy**:
```bash
heroku create tennis-maddpg-demo
git push heroku main
```

## �� Testing Deployment

### Documentation Tests
```bash
# Check documentation builds without errors
sphinx-build -W -b html docs docs/_build/html

# Test links
sphinx-build -b linkcheck docs docs/_build/linkcheck
```

### Web Demo Tests
```bash
# Test demo locally
streamlit run src/web_demo.py --server.headless true

# Run in browser automation
pytest tests/test_web_demo.py
```

## 🔐 Production Considerations

### Security
- Remove debug flags in production
- Use environment variables for sensitive data
- Enable HTTPS for public deployments

### Performance
- Use CDN for static assets
- Enable caching for documentation
- Optimize model loading for web demo

### Monitoring
- Add analytics to documentation
- Monitor web demo usage
- Set up error tracking

## 📋 Deployment Checklist

- [ ] Documentation builds successfully
- [ ] All links work correctly
- [ ] Web demo runs without errors
- [ ] Models load properly
- [ ] Environment variables configured
- [ ] Security settings enabled
- [ ] Performance optimized
- [ ] Monitoring in place
- [ ] Backup strategy defined
- [ ] Update process documented
