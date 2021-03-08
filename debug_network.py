import copy
import numpy as np
from nn_lib import *

def debug_net():
    '''Debug network by calculating gradient and numerical gradient'''

    frst_layer = 60
    hidden = [40, 20]
    out_layer = 10

    # First without regularization
    network = NetworkObject(inpt = frst_layer, hidden = hidden,
            outpt = out_layer, lbda = 0)
    rel_diff = num_grad(network, frst_layer, 1e-4)

    print("Maximum relative difference in numerical gradient ", end = "")
    print("without regularization = {:.2e}".format(rel_diff))

    # Now with regularization
    network = NetworkObject(inpt = frst_layer, hidden = hidden,
            outpt = out_layer, lbda = 0.1)
    rel_diff = num_grad(network, frst_layer, 1e-4)

    print("Maximum relative difference in numerical gradient ", end = "")
    print("with regularization = {:.2e}".format(rel_diff))

def num_grad(network, inpt_siz, eps):
    # Create random input
    nExamples = 100
    lab = 2
    input_arr = np.random.random((nExamples, inpt_siz))

    # Calculate analytical gradient
    network.calc_gradient(input_arr, [lab]*nExamples, nExamples)
    analyt_grad = network.get_gradient()

    # Now calculate numerical gradient
    # Get the thetas and make a copy
    thetas = network.get_thetas()
    thetas_cpy = copy.deepcopy(thetas)

    # For each possible theta, calculate the gradient
    num_grad = []
    for grad in analyt_grad:
        num_grad.append(grad * 0)

    # Go theta by theta adding eps and -eps and calculating cost
    for kk in range(len(num_grad)):
        nn, mm = np.shape(num_grad[kk])
        for ii in range(nn):
            for jj in range(mm):
                # Plus
                thetas_cpy[kk][ii][jj] += eps
                network.set_thetas(thetas_cpy)
                jp = network.get_cost(input_arr, [lab]*nExamples)

                # Minus
                thetas_cpy[kk][ii][jj] -= 2 * eps
                network.set_thetas(thetas_cpy)
                jm = network.get_cost(input_arr, [lab]*nExamples)

                num_grad[kk][ii][jj] = (jp - jm) / (2 * eps)

                # Restore copy
                thetas_cpy[kk][ii][jj] = thetas[kk][ii][jj]

    rel_diff = None
    for kk in range(len(num_grad)):
        diff = np.amax(abs((analyt_grad[kk] - num_grad[kk])/analyt_grad[kk]))
        if rel_diff is None:
            rel_diff = diff
        else:
            rel_diff = max(diff, rel_diff)

    return rel_diff

debug_net()
