# Plotting and analysis code for "Brain-machine interface control with artificial intelligence copilots"
This repository contains code for processing, analysis, and plotting experimental results.
The code for CNN-KF and copilot model training can be found in the `bci_raspy` repository (may be subject to a name change).


## Requirements
Python 3.8.10
Tested on 
Install `pip` packages with `pip install -r requirements.txt`.
After that, run `pip install -e .` from this directory or `setup.py develop` to use this package.


## Directories

The jupyter notebooks in `scripts/` preprocesses data and stores them in `data/`, as well as generates figures based on 
Generated figures are stored in `figs/`.
Subdirectories are organized by the type of experiment and analysis performed, e.g. for CNN-KF adaptation, copilot AB, and robotic arm experiments.
Plotting scripts can be re-run by running all cells of their respective jupyter notebook, e.g. `scripts/adaptation/gen_adaptation_fig.ipynb`.