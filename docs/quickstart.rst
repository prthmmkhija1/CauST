Quickstart
==========

.. code-block:: python

   from caust import CauST
   from caust.data.loader import load_and_preprocess

   adata = load_and_preprocess("data/raw/sample.h5ad")

   model = CauST(n_causal_genes=50, n_clusters=7, epochs=200)
   result = model.fit_transform(adata)

   top_genes = model.get_top_causal_genes(n=20)
   print(top_genes)

See the ``tutorials/`` folder for full worked examples.
