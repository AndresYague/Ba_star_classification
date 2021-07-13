import numpy as np
import matplotlib.pyplot as plt
import os, sys, random
from check_data_lib import *

from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf

def give_inputs_labels(all_models):
    """
    Divide into input and labels
    """

    inputs = []
    labels = []
    label_dict = {}
    ii = 0
    for model in all_models:
        lnlst = model.split()

        # Add the inputs
        inputs.append(np.array(list(map(lambda x: float(x), lnlst[0:-1]))))
        label = lnlst[-1]

        # Number the label
        if label not in label_dict:
            label_dict[label] = ii
            ii += 1

        labels.append(label_dict[label])

    return np.array(inputs), np.array(labels), label_dict

def create_model(train_inputs, train_labels, label_dict, layers=[],
                 mod_dir=None):
    """
    Create or load the model, depending on mod_dir
    """

    # Flags for model existence
    model_exists = False
    created = True

    # Check if this model exists
    if mod_dir is not None:
        model_exists = os.path.isdir(mod_dir)
        created = not model_exists

    # Just load model
    models = []
    if model_exists:

        # Get all models
        directories = get_list_networks(mod_dir)
        for sub_dir in directories:
            models.append(tf.keras.models.load_model(sub_dir))

    # Or create and train it
    else:

        # Numbers for the model
        nn = len(train_inputs[0])
        outpt = len(label_dict)

        # Create the network
        model = tf.keras.Sequential()

        # Input layer
        model.add(tf.keras.layers.Dense(layers[0], input_shape = (nn, ),
                  activation = "relu"))

        # Hidden layers
        for lay in layers[1:]:
            model.add(tf.keras.layers.Dense(lay, activation = "relu"))
            model.add(tf.keras.layers.Dropout(0.3))

        # output layer
        model.add(tf.keras.layers.Dense(outpt, activation = "sigmoid"))

        # Choose alpha
        alpha = 0.0015
        if mod_dir is not None and "fruity" in mod_dir:
            alpha = 0.001

        # Compile
        epochs = 10
        optimizer = tf.keras.optimizers.RMSprop(learning_rate = alpha)
        model.compile(optimizer = optimizer,
                metrics = ["sparse_categorical_accuracy"],
                loss = tf.keras.losses.SparseCategoricalCrossentropy())

        # Train
        model.fit(train_inputs, train_labels, epochs = epochs,
                  validation_split = 0.3)
        models.append(model)

        # Save model
        if mod_dir is not None:
            model.save(mod_dir)

    return models, created

def check_model(models, inputs, labels, conf_threshold=0.75, verbose=False):
    """
    Do a check with the provided inputs and labels
    """

    correct_per_label = [0] * len(labels)
    total_per_label = [0] * len(labels)
    confident_per_label = [0] * len(labels)
    confident_correct_per_label = [0] * len(labels)

    # Predict
    best_prediction, all_predictions = predict_with_networks(models, inputs)

    if verbose:
        for ii in range(len(best_prediction)):

            # Confidence
            conf = np.max(best_prediction[ii]) + 1e-40
            conf /= np.sum(best_prediction[ii]) + 1e-20

            # Check threshold
            if conf > conf_threshold:
                confident_per_label[labels[ii]] += 1

            # Add the total
            total_per_label[labels[ii]] += 1

            # And the correct
            if labels[ii] == np.argmax(best_prediction[ii]):
                correct_per_label[labels[ii]] += 1
                if conf > conf_threshold:
                    confident_correct_per_label[labels[ii]] += 1

        s = "\n" + "=" * 10 + "\n"
        tot = np.sum(total_per_label)
        s += f"Total cases = {tot}\n"

        prop = [divide(x, y) for x, y in
                zip(correct_per_label, total_per_label)]
        acc = divide(np.sum(correct_per_label), np.sum(total_per_label)) * 100
        s += f"The total accuracy is {acc:.2f}%\n"

        prop = [divide(x, y) for x, y in
                zip(confident_per_label, total_per_label)]
        conf = divide(np.sum(confident_per_label), np.sum(total_per_label)) * 100
        s += f"Total confident cases {conf:.2f}%\n"


        prop = [divide(x, y) for x, y in
                zip(confident_correct_per_label, confident_per_label)]
        tot = divide(np.sum(confident_correct_per_label),
                     np.sum(confident_per_label))
        tot = tot * 100
        s += f"Total confidently correct cases {tot:.2f}%\n"

        print(s)

def divide(a, b):
    """
    Safe divide a and b
    """

    try:
        return a/b
    except ZeroDivisionError:
        return 0
    except:
        raise

def create_a_network(inputs, labels, label_dict, train_num, test_num, mod_dir,
                     final_dir=None):

    # Shuffle models
    inpt_labs = list(zip(inputs, labels))
    random.shuffle(inpt_labs)
    inputs, labels = zip(*inpt_labs)

    # Convert into numpy arrays
    inputs = np.array(inputs)
    labels = np.array(labels)

    # Separate
    ii0, iif = 0, train_num
    train_inputs, train_labels = inputs[ii0:iif], labels[ii0:iif]

    ii0, iif = iif, iif + test_num
    test_inputs, test_labels = inputs[ii0:iif], labels[ii0:iif]

    # Hidden layers for model
    layers = [len(label_dict) * 10, len(label_dict) * 10]
    if "fruity" in mod_dir:
        layers = [len(label_dict) * 100]

    # Create model
    models, created = create_model(train_inputs, train_labels, label_dict,
                                   layers=layers, mod_dir=mod_dir)

    # Save label dictionary in network directory
    if final_dir is not None:
        name_dict_file = os.path.join(final_dir,
                                      "label_dict_" + final_dir + ".txt")
    else:
        name_dict_file = os.path.join(mod_dir, "label_dict_" + mod_dir + ".txt")

    # Only save if it did not exist before
    if not os.path.isfile(name_dict_file):
        with open(name_dict_file, "w") as fwrite:
            for key in label_dict:
                fwrite.write(f"{key} {label_dict[key]}\n")

    if created:
        print(2 * "\n")
        print("=============================")
        check_model(models, test_inputs, test_labels, verbose=True)
        print("=============================")
        print(2 * "\n")

    # Clear state
    tf.keras.backend.clear_session()

    return models

def main():
    """Create and train neural network"""

    # Create or load model
    if len(sys.argv) > 1:
        mod_dir = sys.argv[1]
    else:
        sys.exit(f"Use: python3 {sys.argv[0]} <network_name> [n_tries]")

    if len(sys.argv) > 2:
        n_tries = int(sys.argv[2])
    else:
        n_tries = 1

    np.random.seed()

    # Read the data
    data_directory = "data_processing_and_plotting"
    if "fruity" in mod_dir:
        models_file = "processed_models_fruity.txt"
        models_file = os.path.join(data_directory, models_file)
    elif "monash" in mod_dir:
        models_file = "processed_models_monash.txt"
        models_file = os.path.join(data_directory, models_file)

    all_models = []
    with open(models_file, "r") as fread:
        header = fread.readline()
        for line in fread:
            all_models.append(line)

    # Now divide in training, CV and test
    tot_num = len(all_models)
    train_num = int(tot_num * 0.8)
    test_num = tot_num - train_num

    # Report back numbers
    print(f"Total models: {tot_num}")
    print(f"Train models: {train_num}")
    print(f"Test models: {test_num}")

    # Transform models into input and labels
    inputs, labels, label_dict = give_inputs_labels(all_models)

    # Add more features to the inputs
    inputs = modify_input(inputs)

    # Make sure to not create new models
    model_exists = os.path.isdir(mod_dir)
    if model_exists:
        # Read each model and then ensemble
        models = create_a_network(inputs, labels, label_dict, train_num,
                                  test_num, mod_dir)

    else:

        # Make models
        models = []
        for ii in range(n_tries):
            this_dir = mod_dir + f"_{ii}"
            this_dir = os.path.join(mod_dir, this_dir)

            print(f"Creating network {ii + 1}/{n_tries}")
            models += create_a_network(inputs, labels, label_dict, train_num,
                                       test_num, this_dir, final_dir=mod_dir)

    # Check
    print(2 * "\n")
    print("=============================")
    print("Testing ensemble")
    print("=============================")
    check_model(models, inputs, labels, verbose=True)

if __name__ == "__main__":
    main()
