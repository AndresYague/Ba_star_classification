import numpy as np
import matplotlib.pyplot as plt
import struct, os, sys

from numpy import linalg as LA
from PCA_lib import *


def save_eigs(eigval,eigvec,filename):
    '''Save this network in file'''

    with open(filename, "wb") as fwrite:
       np.save(fwrite,eigval)
       for eigv in eigvec:
           np.save(fwrite, eigv)
 

def GetError(model,data):
    '''Calculate distance between model and data
       via Eckhart-Young Theorem and Frobenius Norm.
       Using these LA theorems we calculate a relative error
       that is commonly used to choose k within an uncertainty 
       window (rel error aka dist = 0.05 means 5% error on calc)
       '''       

    # Get relative distance between model and data

    # Absolute values for model and data
    abs_model = np.abs(model)
    abs_data = np.abs(data)

    # Only when model or data > 0 (ignore otherwise)
    non_zero_mod = np.array([x > 1e-10 for x in abs_model])
    non_zero_data = np.array([x > 1e-10 for x in abs_data])

    # Do the or (True + False = True)
    non_zero = non_zero_mod + non_zero_data

    # Get distances
    dist = LA.norm(model-data,'fro')**2/LA.norm(abs_model)**2
    
    # Divide by the total values (so 0 and 0 count as accurate)
    dist = dist / model.shape[1]

    return dist
    
    
def plot_error(err):
    '''Plot the error between original and approximated data '''
    
    plt.figure(0)
    listmu=range(1,len(err)+1,1)
    plt.xlabel('# of eigenvectors')
    plt.ylabel('Error')
    plt.axhline(y=0.05, color='k', linestyle='--')
    plt.plot(listmu,err)
    plt.show()
    

def main():
    """Create PCA framework and plot the outcome to decide on the numbe
    of eigenvectors to include in the classification"""
    
    #Get file names from input
    if len(sys.argv) < 2:
       s = "Incorrect number of arguments. "
       s += f"Use: python3 {sys.argv[0]} <monash or fruity> "
       raise Exception(s)

    if sys.argv[1]=='monash':
        models = 'processed_models_monash.txt'
    if sys.argv[1]=='fruity':
        models = 'processed_models_fruity.txt'    

    all_models = []
    with open(models, "r") as fread:
        header = fread.readline()
        for line in fread:
            all_models.append(line)

    # Transform models into input and labels
    inputs_full, labels, label_dict, labellist = give_inputs_labels(all_models)
    #Option to remove metallicity from PCA by adding: [:,1:]
    inputs=inputs_full #[:,1:]

    # Perform mean normalisation and feature scaling
    # Calculate the eigenvalues and eigenvectors
    eigval,eigvec,mu = get_PCs(inputs)

    # Save the eigval and eigvec
    filename_eigs = 'eigenVs'    
    save_eigs(eigval,eigvec,filename_eigs)

    # Calculate the dim. reduced data, approx data, and distance using K eigenvectors
    err = np.zeros(len(mu))
    for K in range(len(mu)):
       outputs = ProjectData(inputs,eigvec,K)
       approx_inputs = RecoverData(outputs,eigvec,K)
       approx_error = GetError(inputs, approx_inputs)
       err[K] = np.mean(approx_error)
    
    s='Now you can decide on how many eigenvectors you want to include\n' 
    s+='in your PCA classification. We added a dashed line at the 5% error.'
    print(s)
    
    #Plot error on distance between model and approx model vs K eigenvectors
    plot_error(err)


if __name__ == "__main__":
    main()
