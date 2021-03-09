import numpy as np
import matplotlib.pyplot as plt
import struct, os, sys

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

def create_model(train_inputs, train_labels, label_dict, layers = [], mod_dir = None):
    """
    Create or load the model, depending on mod_dir
    """

    # Check if this model exists
    model_exists = False
    if mod_dir is not None:
        model_exists = os.path.isdir(mod_dir)

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
        model.add(tf.keras.layers.Dense(layers[0], input_shape = (nn, ), activation = "relu"))

        # Hidden layers
        for lay in layers[1:]:
            model.add(tf.keras.layers.Dropout(0.1))
            model.add(tf.keras.layers.Dense(lay, activation = "relu"))

        # output layer
        model.add(tf.keras.layers.Dense(outpt, activation = "sigmoid"))

        # Compile
        model.compile(optimizer = "adam",
                metrics = ["sparse_categorical_accuracy"],
                loss = tf.keras.losses.SparseCategoricalCrossentropy())

        # Train
        model.fit(train_inputs, train_labels, epochs = 10000, validation_split = 0.3)

        # Save model
        if mod_dir is not None:
            model.save(mod_dir)

    return model

def main():
    """Create and train neural network"""

    # Create or load model
    if len(sys.argv) > 1:
        mod_dir = sys.argv[1]
    else:
        mod_dir = None

    # Read the data
    shuffled_models = "shuffled_models.txt"
    all_models = []
    with open(shuffled_models, "r") as fread:
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

    # Save label dictionary
    if mod_dir is not None:
        name_dict_file = "label_dict_" + sys.argv[1] + ".txt"
        with open(name_dict_file, "w") as fwrite:
            for key in label_dict:
                fwrite.write(f"{key} {label_dict[key]}\n")

    # Separate
    ii0, iif = 0, train_num
    train_inputs, train_labels = inputs[ii0:iif], labels[ii0:iif]

    ii0, iif = iif, iif + test_num
    test_inputs, test_labels = inputs[ii0:iif], labels[ii0:iif]

    # Hidden layers for model
    outpt = len(label_dict)
    layers = [100, 100, 100, 100]

    # Create model
    model = create_model(train_inputs, train_labels, label_dict, layers = layers, mod_dir = mod_dir)

    # Check
    test_loss, test_acc = model.evaluate(test_inputs, test_labels, verbose = 2)

    # Predict
    predictions = model.predict(test_inputs)

    # Count correct cases
    correct_cases = 0; confident_cases = 0; confident_correct = 0
    for ii in range(len(predictions)):
        conf = max(predictions[ii])/sum(predictions[ii])
        if test_labels[ii] == np.argmax(predictions[ii]):
            correct_cases += 1
            if conf > 0.75:
                confident_correct += 1

        if conf > 0.75:
            confident_cases += 1

    print()
    acc = correct_cases/len(test_labels) * 100
    print("Correct cases = {:.2f}%".format(acc))

    prop = confident_cases/len(test_labels) * 100
    print("Proportion of confident cases = {:.2f}%".format(prop))

    try:
        acc = confident_correct/confident_cases * 100
    except ZeroDivisionError:
        acc = 0
    print("Confidently correct cases = {:.2f}%".format(acc))

if __name__ == "__main__":
    main()
