import sys
import numpy as np
from check_data_lib import *

def find_k(model, data, tol = 1e-3):
    """
    Find the minimum dilution and distance for this model
    """

    # Start with the extremes
    k0 = 0
    k1 = 1
    km = 0.5

    while True:

        # Each of the gradients
        sum_dist0, grad0 = get_one_gradient(model, data, k0)
        sum_distm, gradm = get_one_gradient(model, data, km)
        sum_dist1, grad1 = get_one_gradient(model, data, k1)

        # Check what to change
        if gradm/grad0 < 0:
            k1 = km
        elif gradm/grad1 < 0:
            k0 = km
        else:
            return sum_dist1, k1

        # Get the middle point
        new_km = (k0 + k1) * 0.5

        # Check if converged
        dif = np.abs(new_km - km)/km
        if dif < tol:
            return sum_distm, km

        # Update
        km = new_km

def get_distances_dilutions(data, all_models, tol = 1e-3):
    """
    Get the minimum distance and dilution to each model from each data point
    """

    # For each model
    all_dist = []
    all_dil = []

    # Find minimum distance and dilution
    for model in all_models:
        sum_dist, dil = find_k(model[1:], data[1:], tol = tol)

        # Calculate distance for this dilution
        sum_dist += np.abs((model[0] - data[0]) ** 2 / data[0])
        all_dist.append(sum_dist / len(model))
        all_dil.append(dil)

    return np.array(all_dist), np.array(all_dil)

def get_closest(data, all_models, all_labels):
    """
    Find closest model
    """

    # Get distances and dilutions
    distances, dilutions = get_distances_dilutions(data, all_models, tol = 1e-5)

    # Remove from the list those dilutions above threshold
    threshold = 0.9
    change = True
    while change:

        change = False

        # Get confidences
        confidences = np.maximum(
                1 - np.abs(distances) / np.average(np.abs(distances)), 0)

        # Give index
        index = np.argmax(confidences)

        # Remove element
        if dilutions[index] > threshold and np.min(dilutions) < threshold:

            # Convert to lists
            lst_dst = list(distances)
            lst_dil = list(dilutions)

            # Remove
            lst_dst.pop(index)
            lst_dil.pop(index)

            # Back to arrays
            distances = np.array(lst_dst)
            dilutions = np.array(lst_dil)

            # Flag change
            change = True

    # Print
    s = "Label {} with probability of {:.2f}%".format(all_labels[index], 0)
    s += " dilution {:.2f} average residual {:.2f}".format(dilutions[index],
            distances[index])
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
                all_models.append(model)
                all_labels.append(label)

    return np.array(all_models), all_labels

def main():
    """
    Load models and pass the Ba stars data
    """

    # Get mode
    mode = None
    if len(sys.argv) > 1:
        mode = sys.argv[1]

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
            get_closest(data, models_monash, labels_monash)
            get_closest(data, models_fruity, labels_fruity)
        elif "monash" == mode:
            get_closest(data, models_monash, labels_monash)
        elif "fruity" == mode:
            get_closest(data, models_fruity, labels_fruity)
        else:
            raise Exception("mode must be monash or fruity")

        # Separate for the next case
        print("------")
        print()


if __name__ == "__main__":
    main()
