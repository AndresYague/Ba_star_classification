Software for the Ba-stars classification project, extended to Paper III (Vilagos et al. 2024)

Use of this software:

============================================================================
PROCESS DATA
============================================================================

All processing data scripts are in the data_processing_and_plotting directory

All of the classificators need a first run of process_data.py. To run it:

python3 process_data.py [y/n]

where the last optional argument, "y" or "n" decides whether to dilute the
models (alternatively, you can set it at the beginning of the code).
The models must be diluted for the neural network and random forest training to
work, but not the closest point classificator nor the neural network and random forest
classificator. If nothing is specified, then the models will be diluted.

The element set for process_data.py is taken from element_set.dat.
Currently, "Fe/H" must be the first element present. All the other elements
are optional, but they have to exist in the database.

Once run, this script will create three files:
-processed_models_fruity.txt (or processed_nondil_models_fruity.txt)
-processed_models_monash.txt (or processed_nondil_models_monash.txt)
-processed_data.txt


============================================================================
NEURAL NETWORK CLASSIFICATION
============================================================================

To use the neural network classificator, first the neural network must be
created with create_network. The command is (or give the parameters at the
beginning of the code):

python3 create_network.py <network_name> [n_tries]

network_name is the name for the directory that will save the network. This
name must contain either "fruity" or "monash", which will indicate to the
script what processed models data should be used.

n_tries is the number of networks that will be trained for the ensemble. By
default n_tries = 1 and will be used only if a new network is being trained.

If an existing directory is passed, it will just test it by using the available
data.

Feel free to change any variable in the network. Perhaps one of the most
fundamental is the "layers" variable, which indicates the number of hidden
layers in the network as well as the number of units. The network itself is
specified in the "create_a_network" and "create_model" functions.
Now the output layer is "softmax", which allows for giving a probability for
every model and not only selecting the best-fitting one.

-----------------------------------------------
Finally, to classify with the network just run (or give it at the beginning of
the code):

python3 classify_with_nn.py <network_name>

Now an AGB model is considered as fitting if 'n_above' number of neural networks
identified the models to have probability higher than 'proba_limit'. These two
variables may be modified at the beginning of the code.

The output will be written to stdout.


============================================================================
CLOSEST ELEMENT CLASSIFICATION
============================================================================

To use the closest element classificator, just run

python3 classify_closest.py

Please modify the variable 'below_top_gof' to control how many AGB models should
be accepted. By default, it is 0.05, which means that models that reach at least
the maximal GoF in the set minus 0.05 are classified.

The output will be written to stdout.


============================================================================
RANDOM FOREST CLASSIFICATION
============================================================================

To create the random forest classifiers and/or to classify with them, run

python3 rf.py [mod_dir]

'mod_dir' controls the directory to save the random forest model file (it may
be modified at the beginning of the code instead of input).

The variable 'n_classifier' controls the number of random forests to use,
while 'fit_if_min_classifier' sets the minimal number of classifiers that
require the probability to be above 'probability_limit' in order to classify
the model as fitting. 'order' controls the subtraction order for the abundances.

If the directory exists, the code will just read in the models from the
.joblib files, and create the classification for the Ba stars.
It it does not exist, the code creates the classifiers and the classification.

The classification is saved to a file with the same filename as the directory.
By default, the feature importances plot is also created.

============================================================================
FINAL CLASSIFICATION
============================================================================

Not used in Paper III.


============================================================================
PLOTTING RESULTS: STATISTICS (PAPER III)
============================================================================

All these new plotting routines may be found in the directory plotting_resids_correls.

-- VIOLIN PLOTS: plot_violin.py ------------------
The code creates the violin- and boxplots for all the features in one plot, as well as
separately for the abundances and elemental ratios of different peaks, as in the paper.

The code is capable to either plot only one classifier's result, or to compare
the result of the three classifier like in Fig. B.3. If the comparison plot is to be
made, set the variable 'allthree' to True at the beginning of the file.

Inputs (either in the terminal input or at the beginning of the file)
- 'files': filename(s) with the text files contatining the results of the classification(s)
- 'pathn': output path name in which to save the plots
- 'set': element set to use
- 'monfru': "mon" or "fru" for Monash or FRUITY

-- CORRELATION PLOTS: plot_correl.py ---------------
Creates the plots for all types of correlations. The input parameters can be set at the beginning of the code.
NOTE: this code requires "boxplot_resids.dat" as an input, that is created with the plot_violin.py.
Please run the plot_violins.py with the results of the desired classifiers first.

The variables 'x_axis' and 'y_axis' control the type of quantities plotted on the two axes, respectively.
This creates a family of plots and outputs them in the same directory.

At the end of this code, you find cases for each plot families. All the implemented type of plots
are shown here. If you only want to run a few of the plots in the chosen family, uncomment the others.

If you want to obtain the Nb-Zr thermometer plots (Figs 15 and 16), see the variables 'only_NbZr', 'onlyBa'
and 'Nb_lim'. Their usage is described as comments in the code.

-- DISTRIBUTION PLOTS: plot_distribution.py --------
Creates the plots for the distribution of classified AGB model parameters (mass, delta, GoF).
Please choose the mode with the variable 'distr_mode'. The variable 'dir' controls the output directory.
The input files should be given in the 'files' variable.

Below, you can find variables controlling the output filename, the xlabel, and the title and
other plotting settings.

-- CROSS SECTIONS VS T PLOT: plot_crossec.py -------
Creates the plot for the omega* calculated from the Zr isotopic cross sections as a function of T.
Requires the cross section datafiles.



============================================================================
PLOTTING RESULTS: INDIVIDUAL STARS
============================================================================

This script needs the output from the classificators to be written into a
textfile.

To use just go to the data_processing_and_plotting directory and there run

python3 plot_stars.py <file1> [file2 ...] <directory>

where "file1" is the path to the first file with output to plot and "file2",
"file3", etc are the optional rest. "directory" is the path to the directory
where the figures will be plotted.

The output files can have any combination of models.

The red stars are the elements taken from element_set.dat