import numpy as np
import matplotlib.pyplot as plt
import struct, os, sys
from nn_lib import *

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

def main():
    """Create and train neural network"""

    # Read the data
    shuffled_models = "shuffled_models.txt"
    all_models = []
    with open(shuffled_models, "r") as fread:
        header = fread.readline()
        for line in fread:
            all_models.append(line)

    # Now divide in training, CV and test
    tot_num = len(all_models)
    train_num = int(tot_num * 0.6)
    cv_num = int(tot_num * 0.2)
    test_num = tot_num - train_num - cv_num

    # Report back numbers
    print("Total models: {}".format(tot_num))
    print("Train models: {}".format(train_num))
    print("Cross-validation models: {}".format(cv_num))
    print("Test models: {}".format(test_num))

    # Transform models into input and labels
    inputs, labels, label_dict = give_inputs_labels(all_models)

    # Separate
    ii0, iif = 0, train_num
    train_inptus, train_labels = inputs[ii0:iif], labels[ii0:iif]

    ii0, iif = iif, iif + cv_num
    cv_inputs, cv_labels = inputs[ii0:iif], labels[ii0:iif]

    ii0, iif = iif, iif + test_num
    test_inputs, test_labels = inputs[ii0:iif], labels[ii0:iif]

    # Create the network
    nn = len(inputs[0])
    outpt = len(label_dict)
    hidden = [outpt, outpt, outpt]
    #hidden = [10, 10, 100]
    models_nn = NetworkObject(inpt = nn, hidden = hidden, outpt = outpt,
                                lbda = 1e-5)

    cost = models_nn.train(train_inptus, train_labels, batch_siz = 10,
                           cv_in = cv_inputs, cv_lab = cv_labels,
                           alpha = 5e-1, verbose = True,
                           tol = 1e-8, low_cost = 0.3)
    #cost = models_nn.train(train_inptus, train_labels, batch_siz = 10,
                           #alpha = 5e-1, verbose = True,
                           #tol = 1e-8, low_cost = 0.3)

    # Trained, save thetas
    models_nn.save_network(cost)

    # Let user know that the network is trained
    print("Trained, last cost is {:.4f}".format(cost))

    # Save the dictionary label
    name_dict_file = "label_dict_cost_{:.4f}".format(cost)
    with open(name_dict_file, "w") as fwrite:
        for key in label_dict:
            fwrite.write("{} {}\n".format(key, label_dict[key]))

    # Check network in cv set:
    print("Checking accuracy with cross validation set")
    correct_cases = 0; confident_cases = 0; confident_correct = 0
    lab_net, conf = models_nn.propagate_indx_conf(cv_inputs)
    for ii in range(len(cv_labels)):
        if lab_net[ii] == cv_labels[ii]:
            correct_cases += 1
            if conf[ii] > 0.75:
                confident_correct += 1

        if conf[ii] >= 0.75:
            confident_cases += 1

    acc = correct_cases/len(cv_labels) * 100
    print("Correct cases = {:.2f}%".format(acc))

    acc = confident_correct/confident_cases * 100
    print("Confidently correct cases = {:.2f}%".format(acc))

    # Check network in test set:
    print("Checking accuracy with test set")
    correct_cases = 0; confident_cases = 0; confident_correct = 0
    lab_net, conf = models_nn.propagate_indx_conf(test_inputs)
    for ii in range(len(test_labels)):
        if lab_net[ii] == test_labels[ii]:
            correct_cases += 1
            if conf[ii] > 0.75:
                confident_correct += 1

        if conf[ii] >= 0.75:
            confident_cases += 1

    acc = correct_cases/len(test_labels) * 100
    print("Correct cases = {:.2f}%".format(acc))

    acc = confident_correct/confident_cases * 100
    print("Confidently correct cases = {:.2f}%".format(acc))

if __name__ == "__main__":
    main()
