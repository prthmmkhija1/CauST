Quickstart
==========

Basic usage
-----------

Train CauST on a single spatial transcriptomics slice:

.. code-block:: python

   from caust import CauST
   from caust.data.loader import load_and_preprocess

   adata = load_and_preprocess("data/raw/sample.h5ad")

   model = CauST(n_causal_genes=50, n_clusters=7, epochs=200)
   result = model.fit_transform(adata)

   top_genes = model.get_top_causal_genes(n=20)
   print(top_genes)

Choosing a gene-selection mode
------------------------------

CauST supports three filter modes through ``filter_mode``:

.. code-block:: python

   # Hard top-K filter only
   model = CauST(n_causal_genes=500, filter_mode="filter")

   # Soft reweighting only (keeps all genes)
   model = CauST(filter_mode="reweight")

   # Filter then reweight (DEFAULT — strongest in ablation)
   model = CauST(n_causal_genes=500, filter_mode="filter_and_reweight")

Save and reload a trained model
-------------------------------

.. code-block:: python

   model.save("checkpoints/my_model.pt")

   loaded = CauST.load("checkpoints/my_model.pt")
   labels = loaded.predict(adata)

Multi-slice training
--------------------

To train across multiple DLPFC slices with cross-slice invariance scoring:

.. code-block:: python

   import scanpy as sc

   slices = {
       sid: sc.read_h5ad(f"data/processed/DLPFC/{sid}.h5ad")
       for sid in ["151507", "151508", "151509", "151510"]
   }

   model = CauST(n_causal_genes=500, n_clusters=7, epochs=500)
   model.fit_multi(slices)

See the ``tutorials/`` folder for full worked examples, including STAGATE
integration, cross-slice evaluation, and causal gene exploration.
