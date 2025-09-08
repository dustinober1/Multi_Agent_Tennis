#!/bin/bash
# Build documentation script

echo "Building Multi-Agent Tennis Documentation..."

# Install documentation dependencies
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints myst-parser nbsphinx

# Create docs if not exists
if [ ! -d "docs" ]; then
    mkdir -p docs
fi

cd docs

# Build HTML documentation
echo "Building HTML documentation..."
sphinx-build -b html . _build/html

echo "Documentation built successfully!"
echo "Open docs/_build/html/index.html in your browser"

# Optional: start local server
if command -v python3 &> /dev/null; then
    echo "Starting local documentation server..."
    cd _build/html
    python3 -m http.server 8000 &
    echo "Documentation server running at http://localhost:8000"
fi
