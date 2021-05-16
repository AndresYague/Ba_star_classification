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


def plot_errors_stars(proj_mod,all_names,labellist,eigvec,all_data,all_errors,mm,highlight,num):

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
        elif 'fruity' in labellist[i]:
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
    plt.plot(proj_obs[:,0],proj_obs[:,1],'*',color='m',zorder=15)

    for j in range(len(all_names)):

        if highlight in all_names[j]:
           data = all_data[j]
           errors = all_errors[j]

           # Apply errors to observed data and project it to 2D
           use_data = apply_errors(highlight,data, errors, mm)
           proj_errors = ProjectData(use_data,eigvec,2)

           plt.scatter(proj_obs[1,0],proj_obs[1,1],color='w',label=highlight)
           plt.plot(proj_errors[:,0],proj_errors[:,1],'o',color='r',alpha=0.1,zorder=10)
           plt.plot(proj_obs[j,0],proj_obs[j,1],'o',color='m',zorder=15)
           plt.legend(loc=9)

    plt.title('Black:models, square=1Mo, triagle=2Mo, star=3Mo, dots=higher masses')


def main():
    """
    Load data, eigenvectors and plot the Ba stars data
    """

    #Get file names from input
    if len(sys.argv) < 3:
       s = "Incorrect number of arguments. "
       s += f"Use: python3 {sys.argv[0]} <models> <MC_nn> [highlight a star? like 'BD-142678']"
       sys.exit(s)

    # Get mode
    if sys.argv[1]=='monash':
        models = 'processed_models_monash.txt'
    elif sys.argv[1]=='fruity':
        models = 'processed_models_fruity.txt'

    # Get mm
    mm = int(float(sys.argv[2]))

    #Get star name from input
    highlight = 'NONE'
    if len(sys.argv) > 3:
       highlight = sys.argv[3]
    print(highlight, 'is shown as purple circle, and its errors as red ones')

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

    # For K=2 we can make figures
    Knum=2

    proj_mod = ProjectData(inputs,eigvec,Knum)
    proj_obs = ProjectData(all_data,eigvec,Knum)

    #Save projected stars to file for future reference
    save_proj_obs(proj_obs,all_names,'Proj_obs.txt')

    # plot the observed values with errors (data set of mm randomised points per star)
    plot_errors_stars(proj_mod,all_names,labellist,eigvec,all_data,all_errors,mm,highlight, 3)
    plt.show()


if __name__ == "__main__":
    main()
