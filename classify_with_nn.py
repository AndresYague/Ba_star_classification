import numpy as np
import sys, os
import rf_lib as rfl
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from ann_visualizer.visualize import ann_viz
import pandas as pd

from classify_lib import *

from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf

dirname = "NN1_monash"
proba_limit = 0.0001 # minimum pobability above which the model is accepted
n_above = 3 # number of networks in which the probability should be above proba_limit for the model

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

    # Updated selection criterion for models classified as possible polluter: probability is above a limit in n_above networks
    for modelind in range(len(all_predictions)):
        lab_stars_count_now = all_predictions[modelind].copy()
        lab_stars_count_now[lab_stars_count_now >= proba_limit] = 1 # plus one model is accurate if proba is above this limit
        lab_stars_count_now[lab_stars_count_now < proba_limit] = 0

        if modelind == 0:  # if first classifier, initialize labels and importances
            lab_stars_count = lab_stars_count_now.copy()
        else:        # all predictions count to the average, should be divided by ntry
            lab_stars_count = np.add(lab_stars_count, lab_stars_count_now)

    #index_best = np.argmax(best_prediction, axis=1)[0]

    # Unroll al indices
    #indices = [x[0] for x in np.argmax(all_predictions, axis=2)]
    #indices = np.where(best_prediction > 0.05)[1]
    indices = np.where(lab_stars_count >= n_above)[1]
    probas = best_prediction[0][indices]

    # Return labels
    labels = [label_dict[index] for index in indices]
    #labels_set = set(labels)
    labels = [x for x in zip(labels, probas)]
    labels.sort(key=lambda a: a[1], reverse=True)

    #for label in labels_set:
        #s = f"{label} {labels.count(label)}"
        # print(s)
    return labels

def main():
    """
    Load network and pass the Ba stars data
    """

    #if len(sys.argv) < 2:
    #    sys.exit(f"Use: python3 {sys.argv[0]} <network_ensemble>")

    # Load network
    #dirname = sys.argv[1]
    directories = get_list_networks(dirname)
    networks = []
    for sub_dir in directories:
        networks.append(tf.keras.models.load_model(sub_dir))

    # Load label dictionary
    label_name = "label_dict_" + dirname + ".txt"
    label_dict_file = os.path.join(dirname, label_name)
    label_dict = load_label_dict(label_dict_file)

    # Load nondil models
    nondil_models_file = rfl.models_file(dirname, nondil=True) # Non-diluted models made with the preprocess, for GoF
    df_nondil = rfl.df_reader(nondil_models_file)
    labels_nondil = df_nondil['Label']

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

        # If to plot the structure of the neural network
        # plot_model(networks[0], to_file='NN_model_fru.png')
        # ann_viz(networks[0], title="nn_struct", filename="nn_struct_fru")

        # Do the MC study here
        labels_set = predict_star(networks, data, label_dict)

        # Print out results  # name of current star
        star_instance = StarStat(name, list(data), list(errors))

        for ii in range(len(labels_set)): # calculate GoF
            label = labels_set[ii][0]
            proba = labels_set[ii][1]
            k = labels_nondil[labels_nondil == label].index[0]
            curr_model = np.asfarray((df_nondil.iloc[labels_nondil[labels_nondil == label].index[0]])[:-1])
            pVal, dilution = star_instance.calculate_dilution(curr_model, max_dil=0.9)  # Calculate GoF and dil
            if pVal > 0.1 and dilution < 0.89:
                s = f"Label {label} with goodness of fit {pVal*100:.2f}% and dilution {dilution:.2f} , probability {proba:.2f}"
                print(s)

        # Separate for the next case
        print("------")
        print()

        # Feature importance for NN
        #shap.initjs()
        #for network in networks:
            # explainer = shap.TreeExplainer(network)
            # shap_values = explainer.shap_values(data)
            # shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :])
            # shap.summary_plot(shap_values, data, plot_type="bar")
            # plt.savefig('shap')
            #perm = PermutationImportance(network, random_state=1).fit(X, y)
            #eli5.show_weights(perm, feature_names=X.columns.tolist())

if __name__ == "__main__":
    main()
