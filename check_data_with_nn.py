import numpy as np
import sys, os

from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf

def load_label_dict(label_dict_file):
    """
    Just load the label dictionary
    """

    label_dict = {}
    with open(label_dict_file, "r") as fread:
        for line in fread:
            lnlst = line.split()
            label_dict[int(lnlst[-1])] = lnlst[0]

    return label_dict

def apply_errors(arr, arr_err, nn):
    """
    Apply random errors drawn in a MC fashion
    """

    new_arr = arr + np.random.random((nn, len(arr))) * 2 * arr_err - arr_err

    return new_arr

def apply_dilution(model, kk):
    """
    Apply dilution of kk. This formula only works for heavy elements

    kk is the fraction of the Ba-star envelope that comes from the AGB
    """

    # Just apply the formula to each element except Fe/H
    new_model = model * 1
    new_model[1:] = np.log10((1 - kk) + kk * 10 ** model[1:])

    return new_model

def get_distance(model, data, err):
    """
    Calculate a distance between model and data
    """

    # Limits of uncertainty box
    x0 = data - err
    x1 = data + err

    # Weighted by the inverse of the size of the uncertainty
    #sumInverses = np.sum(1/(x1[1:] - x0[1:]))
    #dist = np.sum((model[1:] - data[1:])**2 / (x1[1:] - x0[1:])) / sumInverses

    # Chi square
    dist = np.average(np.abs((model[1:] - data[1:]) ** 2 / data[1:]))

    return dist

def calculate_dilution(data, err, label, processed_models):
    """
    Calculate best dilution for this label and data
    """

    with open(processed_models, "r") as fread:
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
        raise Exception("Label {} not found".format(label))

    # Now transform into floats and np array
    model = np.array(list(map(lambda x: float(x), model)))

    # Dilute
    dk = 0.001
    dil_fact = np.arange(0, 1 + dk, dk)
    minDist = None; minDil = None
    for kk in dil_fact:
        dilut = apply_dilution(model, kk)

        # Check distance between data and diluted model
        dist = get_distance(dilut, data, err)

        # Save smallest distance
        if minDist is None or dist < minDist:
            minDist = dist
            minDil = kk

    return minDil, minDist

def do_mc_this_star(network, data, errors, name, label_dict, nn,
                    processed_models, maxSize = None):
    """
    Calculate the MC runs for this star to the network
    """

    # Apply errors
    use_data = apply_errors(data, errors, nn)

    # In case size is bearable
    if maxSize is None or maxSize > nn:
        # Propagate
        predictions = network.predict(use_data)

        # Get vectors and confidences
        vect_indx = np.argmax(predictions, axis = 1)
        conf_indx = np.max(predictions, axis = 1)
        conf_indx /= np.sum(predictions, axis = 1) + 1e-20

    # Otherwise
    else:
        # Make sure it is integer
        maxSize = int(maxSize)

        # Initialize
        vect_indx = np.array([])
        conf_indx = np.array([])

        # Divide data
        ii = 0
        while True:

            # Get init and end indices
            init = ii * maxSize
            end = min(init + maxSize, nn)

            # Predict
            predictions = network.predict(use_data[init:end])

            # Save
            vect_indx = np.append(vect_indx, np.argmax(predictions, axis = 1))

            conf = np.max(predictions, axis = 1)
            conf /= np.sum(predictions, axis = 1) + 1e-20
            conf_indx = np.append(conf_indx, conf)

            # Check if end
            if end == nn:
                break

            # Advance ii
            ii += 1

    # Now make a dictionary with all the labels and their weight
    norm_labels = {}; norm_fact = 0
    for index, conf in zip(vect_indx, conf_indx):

        # Retrieve the label
        lab = label_dict[index]

        # Add the weight
        if lab in norm_labels:
            norm_labels[lab] += conf
        else:
            norm_labels[lab] = conf

    # Calculate dilution and adjust weights
    dil_labels = {}
    for key in norm_labels:

        # Calculate dilution for this case
        dilut, resd = calculate_dilution(data, errors, key, processed_models)

        # Save dilution and adjust weights
        dil_labels[key] = (dilut, resd)
        norm_labels[key] /= abs(resd)

    # Print
    norm_fact = sum(norm_labels.values())
    for key in norm_labels:

        # Normalize
        prob = norm_labels[key] / norm_fact
        dilut, resd = dil_labels[key]

        # Skip lower probability
        if prob < 0.1:
            continue

        # Print
        s = "Label {} with probability of {:.2f}%".format(key, prob * 100)
        s += " dilution {:.2f} average residual {:.2f}".format(dilut, resd)
        print(s)

def main():
    """
    Load network and pass the Ba stars data
    """

    if len(sys.argv) < 3:
        print(f"Use: python3 {sys.argv[0]} <network> <nn>")
        return 1

    # Load network
    dirname = sys.argv[1]
    network = tf.keras.models.load_model(dirname)

    # Get number of MC runs varying the parameters of each star
    nn = int(float(sys.argv[2]))

    # Load label dictionary
    label_name = "label_dict_" + dirname + ".txt"
    label_dict_file = os.path.join(dirname, label_name)
    label_dict = load_label_dict(label_dict_file)

    # File with models
    if "fruity" in dirname:
        processed_models = "processed_models_fruity.txt"
    elif "monash" in dirname:
        processed_models = "processed_models_monash.txt"

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

    # Maximum number of predictions at once
    maxSize = 5e5

    # Start
    for ii in range(len(all_names)):
        data = all_data[ii]
        errors = all_errors[ii]
        name = all_names[ii]

        # Print output for this star
        print("For star {}:".format(name))

        # Do the MC study here
        do_mc_this_star(network, data, errors, name, label_dict, nn,
                        processed_models, maxSize = maxSize)

        # Separate for the next case
        print("------")
        print()

if __name__ == "__main__":
    main()
