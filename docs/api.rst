API Reference
=============

Pipeline
--------

The main entry point. ``CauST`` follows a scikit-learn-style API.

.. code-block:: python

   from caust import CauST
   model = CauST(n_causal_genes=500, n_clusters=7, epochs=500)
   result = model.fit_transform(adata)

.. autoclass:: caust.CauST
   :members:
   :undoc-members:

Data
----

Loading and preprocessing spatial transcriptomics data.

.. automodule:: caust.data.loader
   :members:

.. automodule:: caust.data.graph
   :members:

Causal
------

Interventional effect estimation, causal gene scoring, and cross-slice
invariance analysis.

.. automodule:: caust.causal.intervention
   :members:

.. automodule:: caust.causal.scorer
   :members:

.. automodule:: caust.causal.invariance
   :members:

Models
------

Built-in GAT autoencoder and wrappers for external backbones.

.. automodule:: caust.models.autoencoder
   :members:

.. automodule:: caust.models.stagate_wrapper
   :members:

Filter
------

Gene selection and reweighting strategies based on causal scores.

.. code-block:: python

   from caust.filter.gene_filter import apply_gene_selection
   adata_filtered = apply_gene_selection(
       adata, gene_scores, mode="filter_and_reweight", k=500
   )

.. automodule:: caust.filter.gene_filter
   :members:

Evaluate
--------

Clustering metrics for domain identification quality.

.. automodule:: caust.evaluate.metrics
   :members:

Visualize
---------

Plotting utilities for spatial domains, gene scores, and benchmarks.

.. automodule:: caust.visualize.plots
   :members:
