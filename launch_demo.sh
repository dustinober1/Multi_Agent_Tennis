#!/bin/bash
# Launch web demo script

echo "🎾 Launching Multi-Agent Tennis Web Demo..."

# Install demo dependencies
pip install streamlit plotly

# Launch Streamlit app
echo "Starting Streamlit server..."
streamlit run src/web_demo.py --server.port 8501 --server.address localhost

echo "Demo available at http://localhost:8501"
