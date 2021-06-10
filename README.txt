Software for the Ba-stars classification project

Use of the software:

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

The output will be written to stdout.

============================================================================
CLOSEST ELEMENT CLASSIFICATION
============================================================================

To use the closest element classificator, just run

python3 check_data_closest.py <MC_nn> [models]

where <MC_nn> has the same meaning as before, and [models] is an optional
argument that can be either "fruity" or "monash". If [models] is not given,
the classification for all the models will be done. Otherwise, it will done
only for the chosen model set.

The output will be written to stdout.

============================================================================
PCA CLASSIFICATION
============================================================================

To use the PCA classificator, first the PCs have to be calculated with
create_PCA. The command is:

python3 create_PCA.py <name>

where <name> is the name of the processed models data that should be used.
This will create the PCs, save them to a file, and create a figure showing
the error between original and approximated data set. With this figure you
can determine the number of eigenvectors you want to include in the
classification (we suggest to use the lowest number of eigenvectors with an
error below 5%).

To classify with PCA, just run:

python3 classify_wPCA.py <name> <MC_nn> <# of eigenvectors>

where <name> and <MC_nn> have the same meaning as before, and
<# of eigenvectors> is the number of eigenvectors that you want to include.

The output will be written to stdout.

============================================================================
CLOSEST ELEMENT CLASSIFICATION WITH K-MEANS
============================================================================

To use the closest element classificator with k-means, just run

python3 classify_wClosestElementKmeans.py <MC_nn> [models]

where <MC_nn> has the same meaning as before, and [models] is an optional
argument that can be either "fruity" or "monash". If [models] is not given,
the classification for all the models will be done. Otherwise, it will done
only for the chosen model set.

The output will be written to stdout.

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

============================================================================
MAKING A LATEX TABLE WITH RESULTS
============================================================================

This script needs the output from the classificators to be written into a
textfile.

To use just go to data_processing_and_plotting and run:

python3 table_maker.py <file1> [file2 ...]

The latex table with be saved as 'Latex_table_results.tex', which includes
shortened names of the models. The second table, 'names.tex', lists the
long and shortened names of all models. If the names of the files contain
either "fruity" or "monash", then the table will only list the fruity or
monash results, respectively.

============================================================================
PLOTTING PCA VISUALISATION
============================================================================

PCA can also be used to visualise data-sets by reducing the dimensions to 2.
To do this just run:

python3 PCA_plot.py <models> <MC_nn> [highlight a star? like 'BD-142678']

where models and MC_nn have the same meaning as before, and there is an
extra option to highlight one star to show the uncertainty range in 2D.
