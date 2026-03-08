Tutorials
=========

CauST includes five Jupyter notebook tutorials in the ``tutorials/`` directory.
Each notebook is self-contained and can be run after installing CauST and
downloading the DLPFC data.

01 — Quickstart
---------------

`tutorials/01_quickstart.ipynb <https://github.com/prthmmkhija1/CauST/blob/main/tutorials/01_quickstart.ipynb>`_

End-to-end single-slice pipeline: load a DLPFC slice, train CauST, inspect
causal gene scores, and evaluate with ARI/NMI.

02 — Custom Data
----------------

`tutorials/02_custom_data.ipynb <https://github.com/prthmmkhija1/CauST/blob/main/tutorials/02_custom_data.ipynb>`_

Bring your own spatial transcriptomics data. Covers AnnData formatting
requirements, preprocessing, and running CauST on non-DLPFC datasets.

03 — STAGATE / GraphST Integration
-----------------------------------

`tutorials/03_integration_STAGATE.ipynb <https://github.com/prthmmkhija1/CauST/blob/main/tutorials/03_integration_STAGATE.ipynb>`_

Swap the default GAT autoencoder for STAGATE or GraphST as the backbone
encoder. Shows how CauST's causal gene filtering improves external models.

04 — Cross-Slice Evaluation
----------------------------

`tutorials/04_cross_slice_evaluation.ipynb <https://github.com/prthmmkhija1/CauST/blob/main/tutorials/04_cross_slice_evaluation.ipynb>`_

Multi-slice training with IRM-style invariance scoring and Leave-One-Donor-Out
(LODO) validation across the 12 DLPFC sections.

05 — Causal Gene Exploration
-----------------------------

`tutorials/05_causal_gene_exploration.ipynb <https://github.com/prthmmkhija1/CauST/blob/main/tutorials/05_causal_gene_exploration.ipynb>`_

Deep dive into causal gene scores: compare scoring methods (gradient vs
perturbation), visualize intervention effects, and export results.
