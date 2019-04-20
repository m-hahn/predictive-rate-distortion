# Code

## Structure

* `analysis/` contains code for plotting results. All data used for the plots reported in the paper are in the TSV files in the `../results/` folder; you can use the scripts in `analysis` to reporoduce all plots reported in the paper.
* `nprd/` contains code for running NPRD.
* `oce/` contains code for running OCE.
* `figures/` contains figures created by various scripts.
* `collectResults/` contains scripts for collecting the results of runs of NPRD/OCE into summary TSV files, overwriting the TSV files in `../results/`.

The scripts in `nprd` and `oce` can be used to recompute predictive rate-distortion curves reported in the paper, or to compute curves for other processes.
See documentation in these directories for details.

## Prerequisites

* All Python code was written for and tested with Python 2.7.
* All R code was tested in version 3.4.4, but should work across versions.
* NPRD and OCE implementations require Pytorch (<https://pytorch.org/>). Code was tested with Pytorch 0.4.1, but should work with later versions.
* NPRD is much faster on GPU, in particular the experiments on natural language.
* For experiments with natural language, you will additionally need datasets (see below).

## Datasets

* Universal Dependencies Corpora (used for POS-level experiment on English; and for Arabic, Russian): These can be obtained freely from <http://universaldependencies.org/>. Store all treebanks for these languages in a common directory, and adapt the path in `nprd/corpusIterator.py` accordingly. Note that the NYUAD Arabic treebank additionally requires a license, see <https://github.com/UniversalDependencies/UD_Arabic-NYUAD/tree/master> for instructions. All other relevant datasets are free.
* Penn Treebank (English): LDC99T42 (<https://catalog.ldc.upenn.edu/LDC99T42>). Adapt the path in `corpusIterator_PTB.py` accordingly. We use nltk.corpus.ptb to access the data (see <http://www.nltk.org/howto/corpus.html> for documentation).
* Japanese Data: LDC95T8 (<https://catalog.ldc.upenn.edu/LDC95T8>). Adapt the path in `nprd/accessLDC95T8.py` accordingly.
* Chinese Data: LDC2012T05 (<https://catalog.ldc.upenn.edu/LDC2012T05>). Adapt the path in `nprd/accessChineseDependencyTreebank.py` accordingly.

## Applying algorithms to other processes

NPRD is a general-purpose method applicable to any stationary process with finite alphabet.
If you want to apply NPRD to another process:
* Edit the function `readUDCorpus()` in `nprd/corpusIteratorToy.py` to add functionality generating samples from this process.
* You can then pass the name of this processes as the `language` argument to the NPRD script (`nprd/run_NPRD_toy.py`) to generate estimates for the predictive rate-distortion curve.
* Use `collectResults/collectResults_NPRD.py` to overwrite the TSV files in `../results/` with the results of this estimation.
* Adapt `analysis/analyze_NPRD_template.R` by replacing `LANGUAGE` with the name given to the new process. Running the script shoud then produce plots of the estimated curve in `figures/`.

