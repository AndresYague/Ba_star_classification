import numpy as np
import sys, os
from classify_lib import *

from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf

from data_processing_and_plotting.process_data_lib import load_ba_stars

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

def predict_star(networks, data, label_dict):
    """
    Calculate the prediction for this star
    """

    # Modify input so it fits the network training
    use_data = np.array([data])
    use_data = modify_input(use_data)

    # Predict
    best_prediction, all_predictions = predict_with_networks(networks, use_data)
    #index_best = np.argmax(best_prediction, axis=1)[0]

    # Unroll al indices
    indices = [x[0] for x in np.argmax(all_predictions, axis=2)]
    max_vals = [x[0] for x in np.max(all_predictions, axis=2)]

    # Add fake label to dictionary
    new_dict = {key:label_dict[key] for key in label_dict}
    fail_key = len(label_dict.keys())
    label_dict[fail_key] = "Fail"

    # Change indices
    for ii in range(len(indices)):
        if max_vals[ii] < 1e-40:
            indices[ii] = fail_key

    # Return labels
    labels = [label_dict[index] for index in indices]
    labels_set = set(labels)
    for label in labels_set:
        s = f"{label} {labels.count(label)}"
        print(s)

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
    all_data, all_errors, all_names, missing_values = load_ba_stars(data_file)

    # Start
    for ii in range(len(all_names)):
        data = all_data[ii]
        errors = all_errors[ii]
        name = all_names[ii]

        # Print output for this star
        print("For star {}:".format(name))

        # Predict the model
        predict_star(networks, data, label_dict)

        # Separate for the next case
        print("------")
        print()

if __name__ == "__main__":
    main()
