# IctalPlaceCell
This package reproduces the major simulation findings in Yi and Pasdarnavab et al. (in submission). 

**Files**:
- gen_and_save_weights_SPREAD.py - runs "training" simulations and saves the trained networks in a .npz file.
- query_replay_spread_NOTEBOOK.ipynb - a jupyter notebook which walks through the specific simulations needed to reproduce all the simulations presented in the paper. 

Justin, D. Y., Pasdarnavab, M., Kueck, L., Tarcsay, G., & Ewell, L. A. (2024). Interictal spikes during spatial working memory carry helpful or distracting representations of space and have opposing impacts on performance. bioRxiv.

**REQUIRED PACKAGES:**
- numpy
- matplotlib
- nest (neural network simulations, see https://nest-simulator.readthedocs.io/en/latest/index.html)
- scipy
- pywt (for wavelets)

_OPTIONAL PACKAGES:_
- joblib (for parallel computing)
- pyinform (for information theoretic calculations)
