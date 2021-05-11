import numpy as np
from numpy import linalg as LA

from check_data_lib import *

def get_PCs(inputs):
    '''Performing mean normalisation and feature scaling'''

    mu = inputs.mean(0)
    X_norm1 = inputs-mu
    sig = X_norm1.std(0)
    inputs_norm = X_norm1/sig

    '''Find the eigenvalues and eigenvectors'''

    Sigma = (1/len(mu))*np.cov(inputs_norm.T)

    eigval,eigvec = LA.eig(Sigma)

    return eigval, eigvec, mu
    
    
def ProjectData(inputs,eigvec,K):
    '''Calculate the reduced data set'''

    Vec_reduce = eigvec[0:K,:].T
    Z = np.matmul(inputs,Vec_reduce)

    return Z
    

def RecoverData(outputs,eigvec,K):
    '''Approximate original input'''

    Vec_reduce = eigvec[0:K,:].T
    approx_X = np.matmul(outputs,Vec_reduce.T)

    return approx_X  
    
    
def load_eigs(filename):
    '''Load eigenvectors in file'''

    eigs=[]
    with open(filename, "rb") as fread:
       for ii in range(10):
           eigs.append(np.load(fread,allow_pickle=True))

    eigval=np.array(eigs[0])
    eigvec=np.array(eigs[1:])

    return eigval,eigvec    
  
    
def give_inputs_labels(all_models):
    """Divide into input and labels    """

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
    
    
def do_mc_this_star(inputs, data, errors, name, labellist,
                    dict_k_means, nn, file_models, eigvec=None, K=0):
    """
    Calculate the MC runs for this star to the network
    """

    # Apply errors
    use_data = apply_errors(name,data, errors, nn)

    # Project data?
    if K > 0:
        use_data = ProjectData(use_data, eigvec, K)


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

        # Find distance of each star to each mean and return the closest mean
        tup = find_distance_models(keys, use_data, maxSize = maxSize)
        indx_k = tup[0]

        # Convert to int
        indx_k = np.array([int(x) for x in indx_k])

        # Now sort the proj_use_data by closest mean
        ord_indx = np.argsort(indx_k)
        use_data = use_data[ord_indx]
        indx_k = indx_k[ord_indx]

        # Find the distance between star and models in the cluster it's closest to,
        # using the sorted list so one cluster after the other
        init = 0
        for ii in range(1, len(indx_k)):

            # Get this cluster
            if indx_k[ii] != indx_k[ii - 1] or ii == len(indx_k) - 1:
                end = ii

                key = tuple(keys[indx_k[ii - 1]])

                # Calculate distance to models within cluster
                tup = find_distance_models(dict_k_means[key],
                        use_data[init:end], maxSize = maxSize)
                indx, min_dist, sum_dist = tup

                # Add these ordered labels
                labels += [inpt_to_label[tuple(dict_k_means[key][int(i)])]
                           for i in indx]

                # Add new distances 
                sum_dist += np.sum((get_distance(x, use_data[init:end])
                                    for x in keys), axis = 1)

                # Get average distances and find confidence level
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
        threshold=0.9
        if norm_labels[key] >= 0.1:
            # Calculate dilution for this case
            dilut, resd = calculate_dilution(data, key, file_models, upper=threshold)

            # Print
            s = "Label {} with probability of {:.2f}%".format(key,
                                                       norm_labels[key] * 100)
            s += " dilution {:.2f} average residual {:.2f}".format(dilut, resd)
            print(s)
            
            
