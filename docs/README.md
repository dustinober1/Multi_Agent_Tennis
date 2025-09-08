# Documentation

This directory contains the complete documentation for the Multi-Agent Tennis project.

## Building Documentation

### Prerequisites
```bash
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints myst-parser nbsphinx
```

### Build HTML Documentation
```bash
# From project root
./build_docs.sh

# Or manually
cd docs
sphinx-build -b html . _build/html
```

### View Documentation
Open `docs/_build/html/index.html` in your web browser.

## Documentation Structure

- `index.rst` - Main documentation index
- `overview.rst` - Project and algorithm overview
- `api/` - Auto-generated API documentation
- `tutorials/` - Step-by-step guides
- `examples/` - Code examples and use cases

## API Documentation

The API documentation is auto-generated from docstrings in the source code using Sphinx autodoc.

## Contributing to Documentation

1. Update docstrings in source code
2. Add new .rst files for additional content
3. Run build script to regenerate
4. Commit changes to version control

## Documentation Features

- **Auto-generated API docs** from source code docstrings
- **Interactive examples** with Jupyter notebooks  
- **Mathematical notation** with MathJax support
- **Professional theme** with search functionality
- **Cross-references** between modules and functions
