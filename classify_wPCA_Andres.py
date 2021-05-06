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

def load_eigs(filename):
    '''Load eigenvectors in file'''

    eigs=[]
    with open(filename, "rb") as fread:
       for ii in range(10):
           eigs.append(np.load(fread,allow_pickle=True))

    eigval=np.array(eigs[0])
    eigvec=np.array(eigs[1:])

    return eigval,eigvec

def ProjectData(inputs,eigvec,K):
    '''Calculate the reduced data set'''

    Vec_reduce = eigvec[0:int(K),:].T
    Z = np.matmul(inputs,Vec_reduce)

    return Z

def save_proj_obs(proj_obs,all_names,filename):
    ''' Save projected observations with their names'''

    f=open(filename,'w')
    for k,l,m in zip(proj_obs[:,0],proj_obs[:,1],all_names):
        f.write(str(k)+"   "+str(l)+"   "+str(m)+"\n")
    f.close()

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
    #print(data.shape[1],data.shape[0])
    if len_shape == 2:
        dist = np.sqrt(np.sum((model - data)**2, axis = 1)) / data.shape[1]
    elif len_shape == 1:
        dist = np.sqrt(np.sum((model - data)**2)) / data.shape[0]
    else:
        raise NotImplementedError

    return dist

def find_distance_models(inputs, proj_use_data, maxSize = 1e4):
    ''' Calculate the closest model to the star'''

    # Break down proj_use_data in batches of maxSize

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
        end = min(init + maxSize, proj_use_data.shape[0])

        # Now get distances for those indices
        Dists=[]
        for model in inputs:
            Dists.append(get_distance(model, proj_use_data[init:end]))
        Dists = np.array(Dists).T

        # Get the minimum index
        idxMin = np.append(idxMin, np.argmin(Dists, axis = 1))

        # And add the minimum&summed distances to a list
        min_dist = np.append(min_dist, np.min(Dists, axis = 1))
        sum_dist = np.append(sum_dist, np.sum(Dists, axis = 1))

        # Exit at the end
        if end == proj_use_data.shape[0]:
            break

        ii += 1

    return idxMin, min_dist, sum_dist

def calculate_dilution(data, err, label):
    """
    Calculate best dilution for this label and data
    """

    with open(sys.argv[1], "r") as fread:
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

def do_mc_this_star(eigvec, K, inputs, data, errors, name, labellist,
                    dict_k_means, nn):
    """
    Calculate the MC runs for this star to the network
    """

    # Apply errors
    use_data = apply_errors(name,data, errors, nn)

    # Project data
    proj_use_data = ProjectData(use_data, eigvec, K)

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
        tup = find_distance_models(keys, proj_use_data, maxSize = maxSize)
        indx_k = tup[0]

        # Convert to int
        indx_k = np.array([int(x) for x in indx_k])

        # Now sort the proj_use_data by closest mean
        ord_indx = np.argsort(indx_k)
        proj_use_data = proj_use_data[ord_indx]
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
                        proj_use_data[init:end], maxSize = maxSize)
                indx, min_dist, sum_dist = tup

                # Add these ordered labels
                labels += [inpt_to_label[tuple(dict_k_means[key][int(i)])]
                           for i in indx]

                # Add new distances 
                sum_dist += np.sum((get_distance(x, proj_use_data[init:end])
                                    for x in keys), axis = 1)

                # Get average distances and find confidence level
                n_data = end - init
                conf = 1 - min_dist / sum_dist * n_data
                conf_arr = np.append(conf_arr, conf)

                init = end

    else:

        # Get distance
        tup = find_distance_models(inputs, proj_use_data, maxSize = maxSize)
        out_arr, min_dist, sum_dist = tup

        # Calculate the average of the distances and normalize
        n_data = len(proj_use_data)
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
            dilut, resd = calculate_dilution(data, errors, key)

            # Print
            s = "Label {} with probability of {:.2f}%".format(key,
                                                       norm_labels[key] * 100)
            s += " dilution {:.2f} average residual {:.2f}".format(dilut, resd)
            print(s)

def plot_stars_models(proj_mod,proj_obs,all_names,labellist,num=0):
    plt.figure(num)
    for i in range(len(labellist)):
        if 'monash' in labellist[i]: 
           if ('z03' in labellist[i] or 'z01' in labellist[i] or 'z014' in labellist[i]):
                alpset=1
           elif ('z0028' in labellist[i] or 'z007' in labellist[i]):
                alpset=0.6
           if 'monash_m1' in labellist[i]: 
                symbol='s'
           elif 'monash_m2' in labellist[i]:
                symbol='v' 
           elif 'monash_m3' in labellist[i]:
                symbol='*' 
           else:
                symbol='.'
        else: 
           if 'm2_' in labellist[i]:
                alpset=1
           elif 'm3_':
                alpset=0.6
           else:
                alpset=0.2
           if 'ty_m1' in labellist[i]: 
                symbol='s'
           elif 'ty_m2' in labellist[i]:
                symbol='v' 
           elif 'ty_m3' in labellist[i]:
                symbol='*' 
           else:
                symbol='.'        
        plt.plot(proj_mod[i,0], proj_mod[i,1],symbol,color='k',alpha=alpset, zorder=1)
    plt.plot(proj_obs[:,0],proj_obs[:,1],'*',color='m')
    for j in range(len(all_names)):
        if 'HD211594' in all_names[j]:
           plt.plot(proj_obs[j,0],proj_obs[j,1],'o',color='m')
    plt.title('Blue:Monash, Black:Fruity, square=1Mo, triagle=2Mo, star=3Mo, dots=higher masses')


def plot_errors_stars(proj_mod,all_names,labellist,eigvec,all_data,all_errors,mm,num):

    proj_obs = ProjectData(all_data,eigvec,2)

    plt.figure(num)
    for i in range(len(labellist)):
        if 'monash' in labellist[i]: 
           if ('z03' in labellist[i] or 'z01' in labellist[i] or 'z014' in labellist[i]):
                alpset=1
           elif ('z0028' in labellist[i] or 'z007' in labellist[i]):
                alpset=0.6
           if 'monash_m1' in labellist[i]: 
                symbol='s'
           elif 'monash_m2' in labellist[i]:
                symbol='v' 
           elif 'monash_m3' in labellist[i]:
                symbol='*' 
           else:
                symbol='.'
        else: 
           if 'm2_' in labellist[i]:
                alpset=1
           elif 'm3_':
                alpset=0.6
           else:
                alpset=0.2
           if 'ty_m1' in labellist[i]: 
                symbol='s'
           elif 'ty_m2' in labellist[i]:
                symbol='v' 
           elif 'ty_m3' in labellist[i]:
                symbol='*' 
           else:
                symbol='.'                 
        plt.plot(proj_mod[i,0], proj_mod[i,1],symbol,color='k',alpha=alpset, zorder=5)
      
    for j in range(len(all_names)):
        #if 'HD210946' in all_names[j]:
        if 'HD211594' in all_names[j]:
        #if 'BD-142678' in all_names[j]:
        #if 'HD49641' in all_names[j]: #insane star
           data = all_data[j]
           errors = all_errors[j]
           # Apply errors to observed data and project it to 2D
           use_data = apply_errors(data, errors, mm)  
           proj_errors = ProjectData(use_data,eigvec,2)              
           plt.plot(proj_errors[:,0],proj_errors[:,1],'o',color='r',alpha=0.1,zorder=1)
           #plt.hist2d(proj_errors[:,0],proj_errors[:,1],cmin=1,bins=50,cmap='Reds',zorder=1)
           plt.plot(proj_obs[j,0],proj_obs[j,1],'o',color='m',zorder=15)

    plt.title('Blue:Monash, Black:Fruity square=1Mo, triagle=2Mo, star=3Mo, dots=higher masses')

def main():
    """
    Load data, eigenvectors and pass the Ba stars data
    """
    #Get file names from input
    if len(sys.argv) < 3:
       s = "Incorrect number of arguments. "
       s += f"Use: python3 {sys.argv[0]} <processed_models.....txt> <# of eigenvectors>"
       raise Exception(s)
    
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

    # Load the eigval and eigvec
    filename = 'eigenVs'
    eigval,eigvec = load_eigs(filename)

    # Read the diluted model data
    models = sys.argv[1]
    all_models = []
    with open(models, "r") as fread:
        header = fread.readline()
        for line in fread:
            all_models.append(line)
    # Transform models into input and labels
    inputs_full, labels, label_dict, labellist = give_inputs_labels(all_models)
    # Option to remove metallicity from PCA by adding: [:,1:]
    inputs=inputs_full #[:,1:]   

    # Calculate projected data
    all_data_full=np.array(all_data)
    all_data=all_data_full #[:,1:]

    # For K=2 we can make figures
    Knum=2
    proj_mod = ProjectData(inputs,eigvec,Knum)
    proj_obs = ProjectData(all_data,eigvec,Knum)
    save_proj_obs(proj_obs,all_names,'Proj_obs.txt')
    #plot_stars_models(proj_mod,proj_obs,all_names,labellist,num=0)

    # plot the observed values with errors (data set of mm randomised points per star)
    mm=100
    #plot_errors_stars(proj_mod,all_names,labellist,eigvec,all_data,all_errors,mm,3)
    #plt.show()
    #stop

    #Now do the classification with K-eigenvectors (K set by input)
    nn = int(1e5) # number of MC runs varying the parameters of each star
    start = time.time()

    # Do a K-means of the proj_mod with k = sqrt(dim)
    Knum=sys.argv[2]
    proj_mod = ProjectData(inputs,eigvec,Knum)
        
    if nn >= 1e4:

        print("Doing the k-means ...")

        # Instantiate class
        k_means = K_means(proj_mod)

        # Do k-means
        n_k = int(np.sqrt(proj_mod.shape[0]))
        k_means.do_k_means(n_k, tol = 1e-1, attempts = 1)

        # Get dictionary
        dict_k_means = k_means.get_min_dictionary()
        print("Done")

    # Ignore K-means if nn is not too large
    else:
        dict_k_means = None


    for ii in range(len(all_names)):
        data = all_data[ii]
        errors = all_errors[ii]
        name = all_names[ii]

        # Print output for this star
        print("For star {}:".format(name))

        do_mc_this_star(eigvec, Knum, proj_mod, data, errors, name, labellist,
                        dict_k_means, nn)

        # Separate for the next case
        print("------")
        print()
    end = time.time()
    print(end - start)

if __name__ == "__main__":
    main()
