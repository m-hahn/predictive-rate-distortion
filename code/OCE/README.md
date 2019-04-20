# OCE

`oce_ba_toy.py` and `oce_ba_pos.py` run OCE on analytically known (``toy'') processes and POS-level modeling of English. We also supply gradient-descent based implementations (`oce_sgd_toy.py`, `oce_sgd_pos.py`) which we sometimes found numerically more stable, and which perform similarly to Blahut-Arimoto (though they are slower to converge).

For preparation, create symlinks to ../nprd/corpusIterator.py and ../nprd/corpusIteratorToy.py.


