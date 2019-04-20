# Code

## Structure

* `analysis` contains code for plotting results.
* `nprd` contains code for running NPRD.
* `oce` contains code for running OCE.
* `figures` contains figures created by various scripts.
* `collectResults` contains scripts for collecting the results of runs of NPRD/OCE into summary TSV files.

All data used for the plots reported in the paper are in the TSV files in the `../results/` folder; you can use the scripts in `analysis` to reporoduce all plots reported in the paper.
You can use the scripts in `nprd` and `oce` to recompute predictive rate-distortion curves reported in the paper, or to compute curves for other processes.
See documentation in these directories for details.

## Prerequisites

* All Python code was written for and tested with Python 2.7.

* All R code was tested in version 3.4.4, but should work across versions.

* NPRD and OCE implementations require Pytorch.

* NPRD works best on GPU.

## Datasets

* Universal Dependencies Corpora (used for POS-level experiment; Arabic, Russian)

* Penn Treebank (English)

* Japanese Data

* Chinese Data

