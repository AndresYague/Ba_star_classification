import numpy as np
import sys, os
from check_data_lib import *

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

def predict_star(networks, data, label_dict, processed_models, star_name):
    """
    Calculate the prediction for this star
    """

    # Modify input so it fits the network training
    use_data = np.array([data])
    use_data = modify_input(use_data)

    # Predict
    prediction = predict_with_networks(networks, use_data)
    index = np.argmax(prediction)
    label = label_dict[index]

    # Calculate dilution for this case
    dilut, resd = calculate_dilution(data, label, processed_models, upper=0.9)
    s = f"Label {label} with dilution {dilut:.2f} average residual {resd:.2f}"
    print(s)

    # Check goodness of fit TODO
    pVal = goodness_of_fit(star_name, data, errors, diluted_model)
    print(pVal)

def main():
    """
    Load network and pass the Ba stars data
    """

    if len(sys.argv) < 2:
        sys.exit(f"Use: python3 {sys.argv[0]} <network_ensemble>")

    # Load network
    dirname = sys.argv[1]
    directories = get_list_networks(dirname)
    networks = []
    for sub_dir in directories:
        networks.append(tf.keras.models.load_model(sub_dir))

    # Load label dictionary
    label_name = "label_dict_" + dirname + ".txt"
    label_dict_file = os.path.join(dirname, label_name)
    label_dict = load_label_dict(label_dict_file)

    # Directory with data
    dir_data = "data_processing_and_plotting"

    # File with models
    if "fruity" in dirname:
        processed_models = "processed_models_fruity.txt"
    elif "monash" in dirname:
        processed_models = "processed_models_monash.txt"

    # Modify path of files to point to dir_data
    processed_models = os.path.join(dir_data, processed_models)
    data_file = os.path.join(dir_data, "processed_data.txt")

    # Now load Ba stars data
    all_data = []; all_names = []; all_errors = []
    with open(data_file, "r") as fread:
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

        # Do the MC study here
        predict_star(networks, data, label_dict, processed_models, name)

        # Separate for the next case
        print("------")
        print()

if __name__ == "__main__":
    main()
