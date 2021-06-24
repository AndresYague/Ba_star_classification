import numpy as np
import matplotlib.pyplot as plt
import struct, os, sys, random, shutil
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

def create_model(train_inputs, train_labels, label_dict, layers = [],
                 mod_dir = None):
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
    if model_exists:

        model = tf.keras.models.load_model(mod_dir)

        # Print architecture
        model.summary()

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

        # Save model
        if mod_dir is not None:
            model.save(mod_dir)

    return model, created

def check_model(model, inputs, labels, label_dict, conf_threshold = 0.75,
                verbose = False):
    """
    Do a check with the provided inputs and labels
    """

    correct_per_label = [0] * len(labels)
    total_per_label = [0] * len(labels)
    confident_per_label = [0] * len(labels)
    confident_correct_per_label = [0] * len(labels)

    # Predict
    predictions = model.predict(inputs)

    if verbose:
        for ii in range(len(predictions)):

            # Confidence
            conf = np.max(predictions[ii]) + 1e-40
            conf /= np.sum(predictions[ii]) + 1e-20

            # Check threshold
            if conf > conf_threshold:
                confident_per_label[labels[ii]] += 1

            # Add the total
            total_per_label[labels[ii]] += 1

            # And the correct
            if labels[ii] == np.argmax(predictions[ii]):
                correct_per_label[labels[ii]] += 1
                if conf > conf_threshold:
                    confident_correct_per_label[labels[ii]] += 1

        s = "\n" + "=" * 10 + "\n"
        avg = np.average(total_per_label, weights = total_per_label)
        tot = np.sum(total_per_label)
        s += "Average cases per label = {:.2f}\n".format(avg)
        s += f"Total cases = {tot}\n"

        prop = [divide(x, y) for x, y in
                zip(correct_per_label, total_per_label)]
        avg = np.average(prop, weights = total_per_label) * 100
        acc = divide(np.sum(correct_per_label), np.sum(total_per_label)) * 100
        s += "The average accuracy is {:.2f}%\n".format(avg)
        s += "The total accuracy is {:.2f}%\n".format(acc)

        prop = [divide(x, y) for x, y in
                zip(confident_per_label, total_per_label)]
        avg = np.average(prop, weights = total_per_label) * 100
        conf = divide(np.sum(confident_per_label), np.sum(total_per_label)) * 100
        s += "Average confident cases {:.2f}%\n".format(avg)
        s += "Total confident cases {:.2f}%\n".format(conf)


        prop = [divide(x, y) for x, y in
                zip(confident_correct_per_label, confident_per_label)]
        avg = np.average(prop, weights = total_per_label) * 100
        tot = divide(np.sum(confident_correct_per_label),
                     np.sum(confident_per_label))
        tot = tot * 100
        s += "Average confidently correct cases {:.2f}%\n".format(avg)
        s += "Total confidently correct cases {:.2f}%\n".format(tot)

        print(s)

    # Give tensorflow values
    test_loss, test_acc = model.evaluate(inputs, labels, verbose = 2)

    return acc, conf

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
    model, created = create_model(train_inputs, train_labels, label_dict,
                                  layers=layers, mod_dir=mod_dir)

    # Save label dictionary in network directory
    if final_dir is not None:
        name_dict_file = os.path.join(mod_dir, "label_dict_" + final_dir + ".txt")
    else:
        name_dict_file = os.path.join(mod_dir, "label_dict_" + mod_dir + ".txt")
    with open(name_dict_file, "w") as fwrite:
        for key in label_dict:
            fwrite.write(f"{key} {label_dict[key]}\n")

    # Check
    print(2 * "\n", "=============================", 2 * "\n")
    if created:
        print("Checking with training set")
        acc, conf = check_model(model, test_inputs, test_labels, label_dict,
                                verbose = True)
    else:
        print("Checking with whole set")
        acc, conf = check_model(model, inputs, labels, label_dict,
                                verbose = True)

    # Clear state
    tf.keras.backend.clear_session()

    return acc, conf

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
    print("Total models: {}".format(tot_num))
    print("Train models: {}".format(train_num))
    print("Test models: {}".format(test_num))

    # Transform models into input and labels
    inputs, labels, label_dict = give_inputs_labels(all_models)

    # Add more features to the inputs
    inputs = modify_input(inputs)

    # Make sure to not try 5 models if not creating
    model_exists = os.path.isdir(mod_dir)
    if model_exists:
        create_a_network(inputs, labels, label_dict, train_num, test_num,
                         mod_dir)
        return 0

    # Make models
    models_dict = {}
    for ii in range(n_tries):
        this_dir = mod_dir + f"_{ii}"

        acc, conf = create_a_network(inputs, labels, label_dict, train_num,
                                     test_num, this_dir, final_dir=mod_dir)
        models_dict[this_dir] = (acc, conf)

    best = None
    for key in models_dict:
        if best is None:
            best = key
            best_acc = models_dict[key][0]
            best_conf = models_dict[key][1]
            continue

        count = models_dict[key][0] + models_dict[key][1]
        if count > best_acc + best_conf:
            best = key
            best_acc = models_dict[key][0]
            best_conf = models_dict[key][1]

    # Remove all not best
    for key in models_dict:
        if key != best:
            shutil.rmtree(key)

    print(f"Choosing: {best}, acc = {models_dict[best][0]:.2f}%,", end = " ")
    print(f"conf = {models_dict[best][1]:.2f}%")
    os.rename(best, mod_dir)

if __name__ == "__main__":
    main()
