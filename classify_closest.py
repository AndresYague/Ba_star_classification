import sys
import numpy as np
import os
from classify_lib import *
from data_processing_and_plotting.process_data_lib import short_name_generator
from data_processing_and_plotting.process_data_lib import new_names

def get_closest(star_instance, all_models, all_labels, top_n=5):
    """
    Find closest model
    """

    # Find best dilution according to fit
    best_dilutions = []
    for ii in range(len(all_labels)):
        model = all_models[ii]
        label = all_labels[ii]

        pVal, dilution = star_instance.calculate_dilution(model, max_dil=0.9)
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
        index = full_names.index(label)
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

    # Now load Ba stars data
    all_data = []; all_names = []; all_errors = []
    with open(file_data, "r") as fread:
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
