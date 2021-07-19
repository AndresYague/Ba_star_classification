import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from process_data_lib import *

def get_correlation(ratio, star_vals, models_monash, models_fruity, output_file):
    '''
    Get values to plot correlation for a given ratio
    '''

    # Define the ratio keys
    ratio_keys = ratio.split("/")
    ratio_keys = [x + "/Fe" for x in ratio_keys]

    # Read every star
    starKey = None
    monash_l2 = []
    fruity_l2 = []
    ratio_values = []
    print(output_file)
    with open(output_file, "r") as fread:
        for line in fread:
            lnlst = line.split()

            # Get new key if any
            if "star" in line:
                starKey = lnlst[-1][:-1]

                # Get ratio
                vals = star_vals[starKey]                
                try:
                   ratVal = vals[ratio_keys[0]] - vals[ratio_keys[1]]
                except:
                   ratVal = 0
                ratio_values.append(ratVal)

            # Get new L2 sum
            if "monash" in line or "fruity" in line:
                # Get label and dilution
                label = lnlst[1]
                dilution = float(lnlst[-4])

                # Get model
                if "monash" in line:
                    model = models_monash[label]
                    list_ = monash_l2
                elif "fruity" in line:
                    model = models_fruity[label]
                    list_ = fruity_l2
                else:
                    print("Only fruity and monash models")
                    raise NotImplementedError

                # Dilute model
                model = apply_dilution(model, dilution, ratio_keys)

                # Calculate L2
                try:
                   l2sum = sum([(vals[x] - model[x])**2 for x in ratio_keys])
                except:
                   l2sum = 0

                # TODO
                # Just one label per star
                #if len(list_) < len(ratio_values):
                    #list_.append(l2sum)
                #else:
                    #if list_[-1] < l2sum:
                        #list_[-1] = l2sum
                # --------

                # TODO
                # All the labels
                list_.append(l2sum)
                while len(list_) > len(ratio_values):
                    ratio_values.append(ratVal)
                # --------
    #print(ratio_values, monash_l2, fruity_l2)
    return ratio_values, monash_l2, fruity_l2

def main():
    '''
    Give goodness of fit correlation for different ratios
    '''

    all_ratios = [
                  "La/Ce",
                  "Sr/Y",
                  "Y/Zr",
                  "Sr/Zr",
                  ]

    # Check arguments
    if len(sys.argv) < 2:
        s = f"python3 {sys.argv[0]} <output_file> <output_file>"
        sys.exit(s)
    output_file1 = sys.argv[1]
    output_file2 = None
    output_file3 = None
    if len(sys.argv)>2:
       output_file2 = sys.argv[2]     
    if len(sys.argv)==4:
       output_file3 = sys.argv[3]   

    # Define all the directories
    dir_data = "Ba_star_classification_data"
    fruity_mods = "models_fruity"
    monash_mods = "models_monash"
    data_file = "all_abund_and_masses.dat"

    fruity_dir = os.path.join(dir_data, fruity_mods)
    monash_dir = os.path.join(dir_data, monash_mods)
    data_file = os.path.join(dir_data, data_file)

    star_vals = get_data_values(data_file)
    models_monash = get_data_monash(monash_dir)
    models_fruity = get_data_fruity(fruity_dir)

    # Plot all ratios
    for ratio in all_ratios:
        ratio_values1, monash_l21, fruity_l21 = get_correlation(ratio, star_vals,
                                                models_monash, models_fruity,
                                                output_file1)
                                                
        #plt.plot(ratio_values1, monash_l21, "cs", markersize=10,label = "M NN")
        #corr_M1 = np.corrcoef(ratio_values1, monash_l21)[0][1]
        corr_F1 = np.corrcoef(ratio_values1, fruity_l21)[0][1]        
        plt.plot(ratio_values1, fruity_l21, "cs", markersize=10,label = "F NN")

        plt.xlabel(f"[{ratio}]")
        plt.ylabel("Sum of L2 differences")


        
        if output_file3 != None:
           ratio_values2, monash_l22, fruity_l22 = get_correlation(ratio, star_vals,
                                                models_monash, models_fruity,
                                                output_file2)
           ratio_values3, monash_l23, fruity_l23 = get_correlation(ratio, star_vals,
                                                models_monash, models_fruity,
                                                output_file3)
           '''plt.plot(ratio_values2, monash_l22, "kd", markersize=5,label = "M Nei")
           plt.plot(ratio_values3, monash_l23, "rx", markersize=10,label = "M mat")
           corr_M2 = np.corrcoef(ratio_values2, monash_l22)[0][1]
           corr_M3 = np.corrcoef(ratio_values3, monash_l23)[0][1]                     
           plt.title(f"CM NN = {corr_M1:.2f}; CM Nei = {corr_M2:.2f}; CM mat = {corr_M3:.2f}") '''  
           plt.plot(ratio_values2, fruity_l22, "kd", markersize=5,label = "F Nei")
           plt.plot(ratio_values3, fruity_l23, "rx", markersize=10,label = "F mat")            
           corr_F2 = np.corrcoef(ratio_values2, fruity_l22)[0][1]
           corr_F3 = np.corrcoef(ratio_values3, fruity_l23)[0][1]
           plt.title(f"CM NN = {corr_F1:.2f}; CM Nei = {corr_F2:.2f}; CM mat = {corr_F3:.2f}") 
           
        elif output_file3 == None and output_file2 != None:
           ratio_values2, monash_l22, fruity_l22 = get_correlation(ratio, star_vals,
                                                models_monash, models_fruity,
                                                output_file2)

           #plt.plot(ratio_values2, monash_l22, "ro", label = "Monash mat")
           plt.plot(ratio_values2, fruity_l22, "ro", label = "F nei")

           #corr_monash2 = np.corrcoef(ratio_values2, monash_l22)[0][1]
           corr_fruity2 = np.corrcoef(ratio_values2, fruity_l22)[0][1]
           plt.title(f"CM NN = {corr_fruity1:.2f}; CM mat = {corr_fruity2:.2f}") 
                      
        else:
           plt.title(f"CM = {corr_monash1:.2f}")#; CF = {corr_fruity1:.2f}")     

        plt.legend()

        filename = "_".join(ratio.split("/"))
        plt.savefig(f"L2_correlation_{filename}.png")
        plt.show()

if __name__ == "__main__":
    main()
