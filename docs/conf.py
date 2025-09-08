# Sphinx documentation configuration
import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

project = 'Multi-Agent Tennis with MADDPG'
copyright = '2025, Dustin Ober'
author = 'Dustin Ober'
release = '1.0.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
    'sphinx.ext.mathjax',
    'sphinx_autodoc_typehints',
    'myst_parser',
    'nbsphinx',
]

source_suffix = {
    '.rst': None,
    '.md': 'myst_parser',
    '.ipynb': 'nbsphinx',
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
}

nbsphinx_execute = 'never'
nbsphinx_allow_errors = True
