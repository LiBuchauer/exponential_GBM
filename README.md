# exponential_GBM
Simulations, models and parameter estimation procedures accompanying a research project on glioblastoma growth in mice.

The research is presented in an article entitled

___Exponential Growth of Glioblastoma Is Driven by Rapidly Dividing and Migrating Cancer Stem Cells in vivo___

Lisa Buchauer\*, Muhammad Amir Khan\*, Yue Zhuo\*, Chunxuan Shao, Peng Zou, Weijun Feng, Mengran Qian, Gözde Bekki, Charlotte Bunne, Anna Neuerburg, Azer Aylin Acikgöz, Mona Tomaschko, Zhe Zhu, Heike Alter, Katharina Hartmann, Olga Friesen, Klaus Hexel, Thomas Höfer\+, Hai-Kun Liu\+

## Structure of the repository

The codebase consists of three main parts. In the first, three common growth laws are fitted to _in vivo_ tumor growth data collected via bioluminescence imaging (__growth_laws_BLI__). Second, there is a simple tumor growth simulation with different cell migration options (__growth_visualisation__). Third, there is the main model discussed in the manuscript entailing cancer stem cells, proliferating progeny and differentiated tumor cells (__hierarchical_model_prime__). For historical reasons, the model is called "prime" (don't worry about it).

### 1) growth_laws_BLI

Contains a single file with the relevant bioluminescence data of untreated glioblastoma as well as three phenomenological models (exponential growth, linear radial growth and Gompertzian growth) which can be fitted to each individual mouse separately.


### 2) growth_visualisation

 The 3D tumor growth simulation is provided as two different versions, one for running single simulations and visualizing their results (local_version) and one for running a higher number of simulations with one set of parameters and combining their results into summary statistics (cluster_version).  
 The main simulation function  `run_sim_3D()` in `vis3D.py` in the local folder directly gives the option to plot a cut through the simulated tumor and a simple 3D plot simulation data used for the images in figure 1D of the manuscript is provided in a separate folder, although the actual images used there were further modified using the 3D-image software blender. Simulation results are also written to .h5 files.  
 In the cluster version, each simulation result is written to .h5 file and not plotted automatically. A collection of .h5 result files can then be jointly imported using functions in the file `plot_cluster_data.py` and summary timecourses can be plotted. Data used for the manuscript's figure 1C is provided in a separate folder.


 ### 3) hierarchical_model_prime

 Here, model parameters are being estimated from a collection of experimental data using Bayesian principles and MCMC sampling. The files in this folder contribute to this task as follows:  

- The basic ODE model is provided in `tumor_ODE.py`   
- `BLI_prime.py`, `Ki67_prime` and `rest_prime` contain experimental data as well as modifications to the basic model that allow a comparison between these data and their simulated counterparts. These scripts return residuals used for parameter estimation below.  
- Based on these files `bayes_prime.py` runs an MCMC chain using the package `emcee` (https://emcee.readthedocs.io/en/v2.2.1/), plots the results and saves the chain or a sample of it file.
- If single-cell tracing data should additionally be incorporated into the fitting process, `bayes_prime_tracing.py` needs to be run instead which additionally incorporates stochastic simulation results produced by the Cython module defined in `SSA_prime_SPmigration.pyx`.  The files `tracing_prime.py` and `tracing_distributions.py` contain the single-cell tracing data, methods for calculating residuals from them and plotting modeled and experimental clone size distributions.
- Stored MCMC parameter chains can be used to project the model into the experimental space and produce plots of this projection using functions in `predict_params.py`. Specific questions can be put to the parameter chain using functions in `query_params.py`.
- The folder `treatment` predicts the tumor response to chemotherapy and stemness knock-down and compares the result to data graphically.
- The folder `MCMC_figure5` contains the MCMC parameter set used for results shown in figures 5 and 6 of the manuscript.
