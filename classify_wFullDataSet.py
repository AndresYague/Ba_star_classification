import numpy as np
import matplotlib.pyplot as plt
import time, sys

from k_means_lib import *
from check_data_lib import *

def give_inputs_labels(all_models):
    """
    Divide into input and labels
    """

    inputs = []
    labels = []
    label_dict = {}
    labellist = []
    ii = 0
    for model in all_models:
        lnlst = model.split()

        # Add the inputs
        inputs.append(np.array(list(map(lambda x: float(x), lnlst[0:-1]))))
        label = lnlst[-1]
        labellist.append(label)

        # Number the label
        if label not in label_dict:
            label_dict[label] = ii
            ii += 1

        labels.append(label_dict[label])

    return np.array(inputs), np.array(labels), label_dict,labellist

def apply_dilution(model, kk):
    """
    Apply dilution of kk. This formula only works for heavy elements

    kk is the fraction of the Ba-star envelope that comes from the AGB
    """

    # Just apply the formula to each element except Fe/H
    new_model = model * 1
    new_model[1:] = np.log10((1 - kk) + kk * 10 ** model[1:])

    return new_model

def get_distance(model, data):
    """
    Calculate a distance between model and data
    """

    len_shape = len(data.shape)
    if len_shape == 2:
        dist = np.sqrt(np.sum((model - data)**2, axis = 1)) / data.shape[1]
    elif len_shape == 1:
        dist = np.sqrt(np.sum((model - data)**2)) / data.shape[0]
    else:
        raise NotImplementedError

    return dist

def find_distance_models(inputs, use_data, maxSize = 1e4):
    ''' Calculate the closest model to the star+err in reduced dimensions'''

    # Break down Ba star data in batches of maxSize

    # Initialize values
    maxSize = int(maxSize)
    idxMin = np.array([])
    min_dist = np.array([])
    sum_dist = np.array([])

    # And start
    ii = 0
    while True:

        # Get initial and final indices
        init = ii * maxSize
        end = min(init + maxSize, use_data.shape[0])

        # Now get distances for those indices
        Dists=[]
        for model in inputs:
            Dists.append(get_distance(model, use_data[init:end]))
        Dists = np.array(Dists).T

        # Get the minimum index
        idxMin = np.append(idxMin, np.argmin(Dists, axis = 1))

        # And the min_dist_norm
        min_dist = np.append(min_dist, np.min(Dists, axis = 1))
        sum_dist = np.append(sum_dist, np.sum(Dists, axis = 1))

        # Exit at the end
        if end == use_data.shape[0]:
            break

        ii += 1

    return idxMin, min_dist, sum_dist

def calculate_dilution(data, err, label,file_models):
    """
    Calculate best dilution for this label and data
    """

    with open(file_models, "r") as fread:
        # Read header
        fread.readline()

        # Find undiluted model
        model = None
        for line in fread:
            lnlst = line.split()
            if lnlst[-1] == label:
                model = lnlst[0:-1]
            elif model is None:
                continue
            else:
                break

    if model is None:
        raise Exception("Label {} not found".format(key))

    # Now transform into floats and np array
    model = np.array(list(map(lambda x: float(x), model)))

    # Dilute
    dk = 0.001
    dil_fact = np.arange(0, 1 + dk, dk)
    minDist = None; minDil = None
    for kk in dil_fact:
        dilut = apply_dilution(model, kk)

        # Check distance between data and diluted model
        dist = get_distance(dilut, data)

        # Save smallest distance
        if minDist is None or dist < minDist:
            minDist = dist
            minDil = kk

    return minDil, minDist

def do_mc_this_star(inputs, data, errors, name, labellist,
                    dict_k_means, nn, file_models):
    """
    Calculate the MC runs for this star to the network
    """

    # Apply errors
    use_data = apply_errors(name, data, errors, nn)

    # Use dict_k_means
    maxSize = 1e6
    conf_arr = np.array([])
    if dict_k_means is not None:

        # List of labels in order with confidence
        labels = []

        # Create dictionary from inputs to label
        inpt_to_label = {}
        for ii in range(len(labellist)):
            inpt_to_label[tuple(inputs[ii])] = labellist[ii]

        # Get keys from k-means
        keys = np.array(list(dict_k_means.keys()))

        # Find distance to each mean
        tup = find_distance_models(keys, use_data, maxSize = maxSize)
        indx_k = tup[0]

        # Convert to int
        indx_k = np.array([int(x) for x in indx_k])

        # Now for each k_means
        # Sort the use_data with indx_arr as key
        ord_indx = np.argsort(indx_k)
        use_data = use_data[ord_indx]
        indx_k = indx_k[ord_indx]

        # Do a distance per group
        init = 0
        for ii in range(1, len(indx_k)):

            # Get this group
            if indx_k[ii] != indx_k[ii - 1] or ii == len(indx_k) - 1:
                end = ii

                key = tuple(keys[indx_k[ii - 1]])

                # Get new values
                tup = find_distance_models(dict_k_means[key],
                        use_data[init:end], maxSize = maxSize)
                indx, min_dist, sum_dist = tup

                # Add these ordered labels
                labels += [inpt_to_label[tuple(dict_k_means[key][int(i)])]
                           for i in indx]

                # Add new values
                sum_dist += np.sum((get_distance(x, use_data[init:end])
                                    for x in keys), axis = 1)

                # Get average distances and normalize
                n_data = end - init
                conf = 1 - min_dist / sum_dist * n_data
                conf_arr = np.append(conf_arr, conf)

                init = end

    else:

        # Get distance
        tup = find_distance_models(inputs, use_data, maxSize = maxSize)
        out_arr, min_dist, sum_dist = tup

        # Calculate the average of the distances and normalize
        n_data = len(use_data)
        conf_arr = 1 - min_dist / sum_dist * n_data

        # Convert to int
        out_arr = np.array([int(x) for x in out_arr])

    # Now make a dictionary with all the labels and their weight
    norm_labels = {}; norm_fact = 0
    for ii in range(len(conf_arr)):
        conf = conf_arr[ii]

        # Get label
        if dict_k_means is None:
            out = out_arr[ii]
            lab = labellist[out]
        else:
            lab = labels[ii]

        # Apply weight
        if lab in norm_labels:
            norm_labels[lab] += conf
        else:
            norm_labels[lab] = conf

        norm_fact += conf

    # Normalize and calculate dilution
    for key in norm_labels:

        # Normalize
        norm_labels[key] /= norm_fact

        if norm_labels[key] >= 0.1:
            # Calculate dilution for this case
            dilut, resd = calculate_dilution(data, errors, key,file_models)

            # Print
            s = "Label {} with probability of {:.2f}%".format(key,
                                                       norm_labels[key] * 100)
            s += " dilution {:.2f} average residual {:.2f}".format(dilut, resd)
            print(s)

def main():
    """
    Load data and pass the Ba stars data
    """
    nn=1e4
    
    if len(sys.argv) < 2:
        print(f"Use: python3 {sys.argv[0]} <nn (1e4?)> [monash or fruity?]")
        return 1

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
                
       
    '''# Read the diluted model data
    file_models = "processed_models.txt"
    all_models = []
    with open(file_models, "r") as fread:
        header = fread.readline()
        for line in fread:
            all_models.append(line)
    # Transform models into input and labels
    inputs, labels, label_dict, labellist = give_inputs_labels(all_models) '''      
         

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
    if nn >= 1e4:

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

    # Ignore K-means if nn is not too large
    else:
        dict_k_means = None

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
