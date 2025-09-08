Getting Started Guide
====================

This tutorial will guide you through setting up and running the Multi-Agent Tennis project.

Installation
------------

1. **Clone the repository**::

    git clone https://github.com/dustinober1/Multi_Agent_Tennis.git
    cd Multi_Agent_Tennis

2. **Set up the environment**::

    chmod +x setup.sh
    ./setup.sh

3. **Install dependencies**::

    pip install -r requirements.txt

Quick Start
-----------

Running Training
~~~~~~~~~~~~~~~~

To start training the MADDPG agents::

    # Interactive training with Jupyter
    jupyter notebook notebooks/Tennis.ipynb
    
    # Direct training
    python src/maddpg_agent.py

Launching Web Demo
~~~~~~~~~~~~~~~~~~

To launch the interactive web demonstration::

    # Using the launch script
    ./launch_demo.sh
    
    # Or directly with Streamlit
    streamlit run src/web_demo.py

The demo will be available at ``http://localhost:8501``

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

To build this documentation::

    # Using the build script
    ./build_docs.sh
    
    # Or manually
    cd docs
    sphinx-build -b html . _build/html

Understanding the Project
-------------------------

Project Structure
~~~~~~~~~~~~~~~~~

The project is organized as follows::

    Multi_Agent_Tennis/
    ├── src/                    # Source code
    │   ├── maddpg_agent.py    # Core MADDPG implementation
    │   ├── config.py          # Configuration management
    │   ├── web_demo.py        # Interactive web demo
    │   └── ...
    ├── models/                # Trained model weights
    ├── docs/                  # Documentation
    ├── notebooks/             # Jupyter notebooks
    └── tests/                 # Unit tests

Key Components
~~~~~~~~~~~~~~

**MADDPG Agent**: The core implementation featuring:

- Actor networks for policy learning
- Critic networks for value estimation  
- Experience replay buffer
- Target networks for stable training

**Configuration System**: Centralized parameter management with:

- Hyperparameter definitions
- Environment settings
- Training configurations

**Web Demo**: Interactive visualization with:

- Real-time agent gameplay
- Performance analytics
- Hyperparameter exploration

Next Steps
----------

- Explore the :doc:`training_guide` for detailed training instructions
- Check out :doc:`../examples/index` for code examples
- Review the :doc:`../api/modules` for API reference

Troubleshooting
---------------

**Import Errors**
    Ensure all dependencies are installed with ``pip install -r requirements.txt``

**CUDA Issues**
    The code automatically detects CUDA availability. For CPU-only usage, no additional setup is needed.

**Environment Issues**
    Make sure you have the Unity Tennis environment in the correct path as specified in ``config.py``
