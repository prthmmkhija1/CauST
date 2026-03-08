CauST Documentation
====================

**Causal Gene Intervention Framework for Spatial Transcriptomics**

CauST identifies spatially variable genes that are *causally* responsible for
tissue domain structure, rather than merely correlated with it.

Overview
--------

Spatial transcriptomics measures gene expression while preserving tissue
coordinates. Existing methods (STAGATE, GraphST) learn spatial domains from
*all* genes, many of which are noise or confounders. CauST uses Pearl's
do-calculus to estimate the causal effect of each gene on domain assignments,
retaining only genes whose perturbation genuinely disrupts spatial structure.

Key features:

- **Causal gene scoring** — perturbation-based and gradient-based effect
  estimation via interventional do-calculus.
- **Invariance analysis** — IRM-style cross-slice consistency scoring.
- **Three gene-selection modes** — hard filter, soft reweight, or combined
  filter-and-reweight (default, strongest in ablation).
- **Plug-in backbone** — works with the built-in GAT autoencoder or external
  models such as STAGATE and GraphST.
- **LODO validation** — Leave-One-Donor-Out protocol for cross-donor
  generalization evaluation.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   tutorials
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
