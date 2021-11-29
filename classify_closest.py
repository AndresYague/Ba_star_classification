import sys
import numpy as np
import os
from classify_lib import *
from data_processing_and_plotting.process_data_lib import short_name_generator
from data_processing_and_plotting.process_data_lib import new_names
from data_processing_and_plotting.process_data_lib import load_ba_stars

def get_closest(star_instance, all_models, all_labels, top_n=5):
    """
    Find closest model
    """

    # Find best dilution according to fit
    best_dilutions = []
    for ii in range(len(all_labels)):
        model = all_models[ii]
        label = all_labels[ii]

        pVal, dilution = star_instance.calculate_dilution(model, max_dil=0.9,
                                                          min_dil=0.1)
        best_dilutions.append((pVal, dilution, label))

    # Sort
    best_dilutions.sort(reverse=True)

    full_names, short_names = new_names(dir_="data_processing_and_plotting")

    # And print the top_n results
    for result in best_dilutions[:min(top_n, len(best_dilutions))]:

        # Index and label for this model
        pVal, dilution, label = result
        if pVal < 0.5:
            break

        # Search for index
        try:
            index = full_names.index(label)
        except ValueError:
            s = "\n======================================================\n"
            s += "The code just found an error in the model label.\n"
            s += "\nPlease make sure to have run the process_data.py script\n"
            s += "without dilution:\n"
            s += "python3 process_data.py n"
            s += "\n======================================================\n"

            print(s)
            raise
        except:
            raise

        label = short_names[index]

        # Print
        s = f"Label {label} with goodness of fit {pVal * 100:.2f}%"
        s += f" and dilution {dilution:.2f}"
        print(s)

def load_models(*args):
    """
    Load models from several files
    """

    all_models = []
    all_labels = []
    for file_ in args:
        with open(file_, "r") as fread:

            # Skip header
            fread.readline()

            # Read next model
            for line in fread:
                lnlst = line.split()
                model = lnlst[0:-1]
                label = lnlst[-1]

                # Transform to floats
                model = np.array(list(map(lambda x: float(x), model)))

                # Save

                # If unique, add
                if len(all_labels) == 0 or label != all_labels[-1]:
                    all_models.append(model)
                    all_labels.append(label)

                # If diluted, only take last model
                else:
                    all_models[-1] = model

    return np.array(all_models), all_labels

def main():
    """
    Load models and pass the Ba stars data
    """

    if len(sys.argv) < 2:
        sys.exit(f"Use: python3 {sys.argv[0]} <nn> [models]")

    # Get nn
    nn = int(float(sys.argv[1]))

    # Get mode
    mode = None
    if len(sys.argv) > 2:
        mode = sys.argv[2]

    dir_path = "data_processing_and_plotting"

    file_monash = "processed_models_monash.txt"
    file_monash = os.path.join(dir_path, file_monash)

    file_fruity = "processed_models_fruity.txt"
    file_fruity = os.path.join(dir_path, file_fruity)

    file_data = "processed_data.txt"
    file_data = os.path.join(dir_path, file_data)

    # Load all these models
    models_monash, labels_monash = load_models(file_monash)
    models_fruity, labels_fruity = load_models(file_fruity)
    all_data, all_errors, all_names, missing_values = load_ba_stars(file_data)
 
    # Start
    for ii in range(len(all_names)):
        data = all_data[ii]
        errors = all_errors[ii]
        name = all_names[ii]

        # initialize the StarStat
        star_instance = StarStat(name, data, errors, nn=nn)

        # Print output for this star
        print("For star {}:".format(name))

        # Get the closest model in monash and then in fruity
        if mode == "monash" or mode is None:
            get_closest(star_instance, models_monash, labels_monash)
        if mode == "fruity" or mode is None:
            get_closest(star_instance, models_fruity, labels_fruity)

        # Separate for the next case
        print("------")
        print()


if __name__ == "__main__":
    main()
