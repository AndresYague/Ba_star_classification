import sys, os
import matplotlib as plt
import numpy as np
from process_data_lib import *

FIGSIZE = [60, 16]
ERRBCOLOR = "yellow"
ERREDGECOLOR = "midnightblue"
PMIN = 2

def get_clean_lnlst(line):
    """
    Clean the input data so it's easier to handle
    """

    # Split line
    lnlst = line.split()

    # Return proper value
    if "Label" in line:
        return [lnlst[1], lnlst[7]]
    elif "star" in line:
        return lnlst
    else:
        return None

def get_dict_predicted(files):
    """
    Make input dictionary to combine all files
    """

    # Initialize dictionary
    dict_ = {}
    dict_["fruity"] = {}
    dict_["monash"] = {}

    # Keep an inventory of repeated models
    repeated = {}
    repeated["fruity"] = {}
    repeated["monash"] = {}

    # Go for file and line
    for file_ in files:
        with open(file_, "r") as fread:

            for line in fread:
                lnlst = get_clean_lnlst(line)

                # Skipe lines without content
                if lnlst is None:
                    continue

                # Read the star name and add the sets in
                # the dictionaries if they were not there
                if "star" in lnlst:
                    star_name = lnlst[-1][:-1]
                    if star_name not in dict_["fruity"]:

                        # The starname set
                        dict_["fruity"][star_name] = set()
                        dict_["monash"][star_name] = set()

                        # Add the list to the repeated models
                        repeated["fruity"][star_name] = []
                        repeated["monash"][star_name] = []

                # Add this line in fruity or monash
                else:
                    if "fruity" in lnlst[0]:
                        type_ = "fruity"
                    elif "monash" in lnlst[0]:
                        type_ = "monash"

                    # Check if repeated to skip
                    if lnlst[0] in repeated[type_][star_name]:
                        continue

                    # Add this model here to avoid repeating it
                    repeated[type_][star_name].append(lnlst[0])

                    # Add to the set
                    dict_[type_][star_name].add(tuple(lnlst))

    return dict_

def plot_results(predicted_models_dict, fruity_models_dict,
                 monash_models_dict, dict_data, red_elements):
    """
    Plot the results, star by star.
    """

    # Grab the data
    name_z = np.loadtxt(os.path.join("data_for_plot", "atomic_nums.dat"),
                        dtype=str)

    # Make an x-array from this information
    start = int(name_z[1][1])
    end = int(name_z[-1][1])
    x_arr = np.arange(start, end)

    # Each key in dict_data is a star
    for key in dict_data:

        # Start the new plot
        # TODO

        # Plot the fruity models
        # TODO

        # Plot the monash models
        # TODO

        # Save the plot
        # TODO

        print(key)

def main():
    """
    Program for plotting the outputs from the classification
    """

    if len(sys.argv) < 3:
        s = "Incorrect number of arguments. "
        s += f"Use: python3 {sys.argv[0]} <file1> [file2 ...] <directory>"
        sys.exit(s)

    # Save files with data and directory
    files = sys.argv[1:-1]
    pathn = sys.argv[-1]

    # Define all the directories
    dir_data = "Ba_star_classification_data"
    fruity_mods = "models_fruity"
    monash_mods = "models_monash"
    data_file = "all_abund_and_masses.dat"

    fruity_dir = os.path.join(dir_data, fruity_mods)
    monash_dir = os.path.join(dir_data, monash_mods)
    data_file = os.path.join(dir_data, data_file)

    # Sort files in fruity and monash
    predicted_models_dict = get_dict_predicted(files)

    # Load the stellar abundances
    dict_data = get_data_values(data_file)

    # The fruity and monash models
    fruity_models_dict = get_data_fruity(fruity_dir)
    monash_models_dict = get_data_monash(monash_dir)

    # Load the red elements
    with open("element_set.dat", "r") as fread:
        for line in fread:
            lnlst = line.split()

            # Skip comments and empty lines
            if len(lnlst) == 0 or "#" in lnlst[0]:
                continue

            red_elements = lnlst
            break

    # Remove "/Fe"
    red_elements = [elem.split("/")[0] for elem in red_elements]
    red_elements.remove("Fe")

    # Now that all the data is loaded, we can plot it
    plot_results(predicted_models_dict, fruity_models_dict,
                 monash_models_dict, dict_data, red_elements)

if __name__ == "__main__":
    main()
