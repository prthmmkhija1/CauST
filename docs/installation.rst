Installation
============

Requirements
------------

- Python >= 3.9
- PyTorch >= 1.12
- PyTorch Geometric
- scanpy, anndata, scikit-learn, scipy, matplotlib

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/prathammakhija/CauST.git
   cd CauST
   pip install -e .

This installs CauST in editable mode along with all required dependencies
listed in ``setup.py``.

Install with development extras
-------------------------------

To include testing and documentation tools:

.. code-block:: bash

   pip install -e ".[dev]"

Optional: STAGATE / GraphST
----------------------------

CauST can use STAGATE or GraphST as drop-in backbone encoders. These are
optional and not required for the default GAT autoencoder pipeline.

.. code-block:: bash

   pip install STAGATE-pyG
   pip install graphst

.. note::

   STAGATE-pyG may not be available on PyPI at all times. If installation
   fails, CauST will still work using its built-in autoencoder.

PyTorch Geometric installation
------------------------------

PyTorch Geometric requires matching CUDA/CPU wheels. Follow the official
guide at https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
or install CPU-only wheels with:

.. code-block:: bash

   pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
       -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)").html
   pip install torch-geometric

Troubleshooting
---------------

**ImportError: No module named torch_geometric**
  Install PyTorch Geometric following the section above.

**CUDA out of memory**
  Reduce ``epochs`` or ``n_clusters``, or run on CPU by passing ``device="cpu"``
  to the ``CauST`` constructor.

**No processed data found**
  Run ``python scripts/01_download_data.py`` and ``python scripts/02_preprocess.py``
  before training scripts.
