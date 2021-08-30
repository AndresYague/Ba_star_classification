Software for the Ba-stars classification project

Use of this software:

============================================================================
PROCESS DATA
============================================================================

All processing data scripts are in the data_processing_and_plotting directory

All of the classificators need a first run of process_data.py. To run it:

python3 process_data.py [y/n]

where the last optional argument, "y" or "n" decides whether to dilute the
models. The models must be diluted for the neural network training to
work, but not the closest point classificator nor the neural network
classificator. If nothing is specified, then the models will be diluted.

The element set for process_data.py is taken from element_set.dat.
Currently, "Fe/H" must be the first element present. All the other elements
are optional, but they have to exist in the database.

# TODO
Allow for Fe/H to be optional or not first

Once run, this script will create three files:
-processed_models_fruity.txt
-processed_models_monash.txt
-process_data.txt

============================================================================
NEURAL NETWORK CLASSIFICATION
============================================================================

To use the neural network classificator, first the neural network must be
created with create_network. The command is:

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

Finally, to classify with the network just run:

python3 classify_with_nn.py <network_name>

The output will be written to stdout.

============================================================================
CLOSEST ELEMENT CLASSIFICATION
============================================================================

To use the closest element classificator, just run

python3 classify_closest.py <MC_nn> [models]

where <MC_nn> has the same meaning as before, and [models] is an optional
argument that can be either "fruity" or "monash". If [models] is not given,
the classification for all the models will be done. Otherwise, it will done
only for the chosen model set.

The output will be written to stdout.

============================================================================
FINAL CLASSIFICATION
============================================================================

TODO

============================================================================
PLOTTING RESULTS
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
