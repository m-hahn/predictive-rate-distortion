# Code

## Structure

* `analysis` contains code for plotting results. All data used for the plots reported in the paper are in the TSV files in the `../results/` folder; you can use the scripts in `analysis` to reporoduce all plots reported in the paper.
* `nprd` contains code for running NPRD.
* `oce` contains code for running OCE.
* `figures` contains figures created by various scripts.
* `collectResults` contains scripts for collecting the results of runs of NPRD/OCE into summary TSV files.

You can use the scripts in `nprd` and `oce` to recompute predictive rate-distortion curves reported in the paper, or to compute curves for other processes.
See documentation in these directories for details.

## Prerequisites

* All Python code was written for and tested with Python 2.7.
* All R code was tested in version 3.4.4, but should work across versions.
* NPRD and OCE implementations require Pytorch. Code was tested with Pytorch 0.4.1, but should work with later versions.
* NPRD is much faster on GPU, in particular the experiments on natural language.

## Datasets

* Universal Dependencies Corpora (used for POS-level experiment; Arabic, Russian): These can be obtained from http://universaldependencies.org/. Store all treebanks for these languages in a common directory, and adapt the path in `nprd/corpusIterator.py` accordingly.

* Penn Treebank (English): LDC99T42 (https://catalog.ldc.upenn.edu/LDC99T42). Adapt the path in `corpusIterator_PTB.py` accordingly. We use nltk.corpus.ptb to access the data (see http://www.nltk.org/howto/corpus.html for documentation).

* Japanese Data: LDC95T8 (https://catalog.ldc.upenn.edu/LDC95T8). Adapt the path in `nprd/accessLDC95T8.py` accordingly.

* Chinese Data: LDC2012T05 (https://catalog.ldc.upenn.edu/LDC2012T05). Adapt the path in `nprd/accessChineseDependencyTreebank.py` accordingly.

