Software for the Ba-stars classification project

Use of the software:

============================================================================
PROCESS DATA
============================================================================

All of the classificators need a first run of process_data.py. To run it:

python3 process_data.py [y/n]

where the last optional argument, "y" or "n" decides whether to dilute the
models. The models must be diluted for the neural network classificator to
work, but not the closest point classificator. If nothing is specified, then
the models will be diluted.

Inside of process_data.py a set of elemental names in "names" is given.
Currently, "Fe/H" must be the first element present. All the other elements
are optional, but they have to exist in the database.

Once run, this script will create three files:
-processed_models_fruity.txt
-processed_models_monash.txt
-process_data.txt

============================================================================
NEURAL NETWORK CLASSIFICATION
============================================================================

To use the neural network classificator, first the neural network must be
created with create_network. The command is:

python3 create_network.py <network_name>

network_name is the name for the directory that will save the network. This
name must contain either "fruity" or "monash", which will indicate to the
script what processed models data should be used.

If an existing model is passed, it will just test it by using the available
data.

Feel free to change any variable in the model. Perhaps one of the most
fundamental is the "layers" variable, which indicates the number of hidden
layers in the model as well as the number of units. The model itself is
specified in the "create_model" function.

Finally, to classify with the network just run:

python3 check_data_with_nn.py <network_name> <MC_nn>

where <MC_nn> is the number of runs to explore the uncertainty values for the
classification. If MC_nn == 0, then errors are not applied.

============================================================================
CLOSEST ELEMENT CLASSIFICATION
============================================================================

To use the closest element classificator, just run

python3 check_data_closest.py <MC_nn> [models]

where <MC_nn> has the same meaning as before, and [models] is an optional
argument that can be either "fruity" or "monash". If [models] is not given,
the classification for all the models will be done. Otherwise, it will done
only for the chosen model set.

============================================================================
PLOTTING DATA
============================================================================

TODO
