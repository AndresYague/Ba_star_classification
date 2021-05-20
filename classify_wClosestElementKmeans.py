import numpy as np
import matplotlib.pyplot as plt
import time, sys

from k_means_lib import *
from check_data_lib import *
from PCA_lib import *


def main():
    """
    Load data and pass the Ba stars data
    """
    nn=1e4

    if len(sys.argv) < 2:
        s= "Incorrect number of arguments. "
        s+= f"Use: python3 {sys.argv[0]} <nn (1e5?)> [monash? fruity? or leave empty]"
        sys.exit(s)

    # Get nn
    nn = int(float(sys.argv[1]))

    # Get mode
    mode = None
    if len(sys.argv) > 2:
        mode = sys.argv[2]

    # Read the model data
    file_M = "processed_models_monash.txt"
    all_models = []
    with open(file_M, "r") as fread:
         header = fread.readline()
         for line in fread:
             all_models.append(line)
         # Transform models into input and labels
         inputs_M, labels_M, label_dict_M, labellist_M = give_inputs_labels(all_models)


    file_F = "processed_models_fruity.txt"
    all_models = []
    with open(file_F, "r") as fread:
         header = fread.readline()
         for line in fread:
             all_models.append(line)
         # Transform models into input and labels
         inputs_F, labels_F, label_dict_F, labellist_F = give_inputs_labels(all_models)


    # Load Ba stars data
    all_data = []; all_names = []; all_errors = []
    with open("processed_data.txt", "r") as fread:
        header = fread.readline().split()[1:]
        for line in fread:
            lnlst = line.split()
            all_names.append(lnlst[-1])

            # Put values in an array
            arr = []; arr_err = []
            for ii in range(len(header)):
                # Skip name
                if "Name" in header[ii]:
                    continue

                # Transform to float
                try:
                    val = float(lnlst[ii])
                except ValueError:
                    val = 1
                except:
                    raise

                # Put errors and values in different arrays
                if "_err" in header[ii]:
                    arr_err.append(val)
                else:
                    arr.append(val)

            # Convert to numpy arrays
            arr = np.array(arr)
            arr_err = np.array(arr_err)

            # Save data
            all_data.append(arr)
            all_errors.append(arr_err)

    #Prep for MC
    #nn = int(1e5) # number of MC runs varying the parameters of each star
    start = time.time()

    # Do a K-means of the proj_mod with k = sqrt(dim)

    print("Doing the k-means ...")

    # Instantiate class monash
    k_means = K_means(inputs_M)

    # Do k-means
    n_k = int(np.sqrt(len(inputs_M)))
    k_means.do_k_means(n_k, tol = 1e-1, attempts = 1)

    # Get dictionary
    dict_k_means_M = k_means.get_min_dictionary()
    print("Done M")

    # Instantiate class fruity
    k_means = K_means(inputs_F)

    # Do k-means
    n_k = int(np.sqrt(len(inputs_F)))
    k_means.do_k_means(n_k, tol = 1e-1, attempts = 1)

    # Get dictionary
    dict_k_means_F = k_means.get_min_dictionary()
    print("Done F")

    for ii in range(len(all_names)):
        data = all_data[ii]
        errors = all_errors[ii]
        name = all_names[ii]

        # Print output for this star
        print("For star {}:".format(name))
        if mode is None:
           do_mc_this_star(inputs_M, data, errors, name, labellist_M,
                        dict_k_means_M, nn, file_M)
           do_mc_this_star(inputs_F, data, errors, name, labellist_F,
                        dict_k_means_F, nn, file_F)

        elif mode=='monash':
           do_mc_this_star(inputs_M, data, errors, name, labellist_M,
                        dict_k_means_M, nn, file_M)

        elif mode=='fruity':
           do_mc_this_star(inputs_F, data, errors, name, labellist_F,
                        dict_k_means_F, nn, file_F)

        else:
            raise Exception("mode must be monash or fruity")

        # Separate for the next case
        print("------")
        print()
    end = time.time()
    print(end - start)

if __name__ == "__main__":
    main()
