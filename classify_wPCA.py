import numpy as np
import matplotlib.pyplot as plt
import time, sys

from k_means_lib import *
from check_data_lib import *
from PCA_lib import *
   

def save_proj_obs(proj_obs,all_names,filename):
    ''' Save projected observations with their names'''

    f=open(filename,'w')
    for k,l,m in zip(proj_obs[:,0],proj_obs[:,1],all_names):
        f.write(str(k)+"   "+str(l)+"   "+str(m)+"\n")
    f.close()


def main():
    """
    Load data, eigenvectors and pass the Ba stars data
    """
    #Get file names from input
    if len(sys.argv) < 3:
       s = "Incorrect number of arguments. "
       s += f"Use: python3 {sys.argv[0]} <monash or fruity> <MC nn> <# of eigenvectors>"
       sys.exit(s)
       
    if sys.argv[1]=='monash':
        models = 'processed_models_monash.txt'
    elif sys.argv[1]=='fruity':
        models = 'processed_models_fruity.txt' 
        
    nn = int(float(sys.argv[2]))
    
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

    # Read the model data
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
    start = time.time()

    # Do a K-means of the proj_mod with k = sqrt(dim)
    Knum=int(sys.argv[3])
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

        do_mc_this_star(proj_mod, data, errors, name, labellist,
                        dict_k_means, nn, models, eigvec, Knum)

        # Separate for the next case
        print("------")
        print()
    end = time.time()
    print(end - start)

if __name__ == "__main__":
    main()
