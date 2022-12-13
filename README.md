# filter-generation

This repository is organised as follows:
- [`baselines`]: 
- [`data_processing`]: All the code to generate data for the filters resides here. Run `data_production.py` to generate filter data and save it locally. 
- [`modeling`]: This folder contains all the code to train the generative models (e.g. VAE, GAN, GMM, etc.) required to run experiments. `_joint` refers to a joint modeling technique. Within each file, modify the `filterpath` and `savepath` according to where your filters are stored, and where you wish to store model checkpoints. 
- [`experiments`]: This folder contains code to run downstream tasks on the MNIST dataset, and there exists a 1-1 correspondence (almost) between the files present here and in the modeling section. Load the trained model (stored in `loadpath`), and run experiments. Results will be saved in a pickle file in `savepath`. 
- [`visualization`]: This folder contains code (in jupyter notebooks) to generate filter samples and histograms for each approach. 
