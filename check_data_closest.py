import sys
import numpy as np
from check_data_lib import *

def get_short_distances(all_data, model, tol = 1e-3):
    """
    Get the minimum distance to this model from each data point
    """

    # Remove metallicity
    all_datat = all_data.T
    data_no_metal = all_datat[1:].T

    # Calculate dilutions
    all_dil = find_k(model[1:], data_no_metal, tol = tol)

    # Dilute
    dil_model = np.array(
            [np.log10((1 - all_dil) + all_dil * 10 ** mod) for mod in model[1:]]
            )

    # Distance to everything but iron (diluted)
    all_dist = np.sum((dil_model.T - data_no_metal) ** 2, axis = 1)

    # Distance to iron
    all_dist += (model[0] - all_datat[0]) ** 2

    # Normalize
    all_dist /= all_data.shape[1]

    return all_dist

def get_closest(data, errors, nn, all_models, all_labels, star_name):
    """
    Find closest model
    """

    # Apply errors
    use_data = apply_errors(star_name, data, errors, nn)

    # Get distances and dilutions
    all_distances = []
    all_dilutions = []
    for model in all_models:
        distances = get_short_distances(use_data, model)
        all_distances.append(distances)

    # Save the numpy arrays
    distances = np.array(all_distances).T

    # Get confidences
    abs_dist = np.abs(distances)
    confidences = np.maximum(1 - abs_dist / np.mean(abs_dist), 0)

    # Give best indices
    indices = np.argmax(confidences, axis = 1)

    # Assign
    label_weight = [0] * len(all_labels)
    for ii, index in enumerate(indices):
        label_weight[index] += confidences[ii][index]

    # Normalize
    label_weight /= np.sum(label_weight)

    # And print
    threshold = 0.9
    for ii, lab in enumerate(all_labels):

        if label_weight[ii] > 0.1:
            # Get dilution and distance
            dilution, dist = calculate_dilution(data, all_models[ii],
                                                upper = threshold)

            # Print
            s = f"Label {lab} with probability of"
            s += " {:.2f}%".format(label_weight[ii] * 100)
            s += " dilution {:.2f}".format(dilution)
            s += " average residual {:.2f}".format(dist)
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

    file_monash = "processed_models_monash.txt"
    file_fruity = "processed_models_fruity.txt"

    # Load all these models
    models_monash, labels_monash = load_models(file_monash)
    models_fruity, labels_fruity = load_models(file_fruity)

    # Now load Ba stars data
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

    # Start
    for ii in range(len(all_names)):
        data = all_data[ii]
        errors = all_errors[ii]
        name = all_names[ii]

        # Print output for this star
        print("For star {}:".format(name))

        # Get the closest model in monash and then in fruity
        if mode is None:
            get_closest(data, errors, nn, models_monash, labels_monash, name)
            get_closest(data, errors, nn, models_fruity, labels_fruity, name)
        elif "monash" == mode:
            get_closest(data, errors, nn, models_monash, labels_monash, name)
        elif "fruity" == mode:
            get_closest(data, errors, nn, models_fruity, labels_fruity, name)
        else:
            raise Exception("mode must be monash or fruity")

        # Separate for the next case
        print("------")
        print()


if __name__ == "__main__":
    main()
