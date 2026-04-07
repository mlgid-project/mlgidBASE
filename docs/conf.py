import os
import sys
import logging
import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib").setLevel(logging.ERROR)

sys.path.insert(0, os.path.abspath('..'))

project = 'mlgidBASE'
author = 'Ainur Abukaev'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'nbsphinx',
    'myst_nb',
]

templates_path = ['_templates']
exclude_patterns = ['_build', '**.ipynb_checkpoints']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

tutorial_dir = os.path.join(os.path.dirname(__file__), 'tutorials')
tutorials = sorted(f for f in os.listdir(tutorial_dir) if f.endswith('.ipynb'))
tutorial_names = [os.path.splitext(f)[0] for f in tutorials]

toctree_path = os.path.join(os.path.dirname(__file__), 'tutorials_toctree.rst')
with open(toctree_path, 'w', encoding='utf-8') as f:
    f.write("Tutorials\n=========\n\n.. toctree::\n   :maxdepth: 2\n\n")
    for name in tutorial_names:
        f.write(f"   tutorials/{name}\n")
