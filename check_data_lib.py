import numpy as np

def apply_errors(arr, arr_err, nn):
    """
    Apply random errors drawn in a MC fashion
    """

    new_arr = arr + np.random.random((nn, len(arr))) * 2 * arr_err - arr_err

    return new_arr

def apply_dilution(model, kk, ignoreFirst = False):
    """
    Apply dilution of kk. This formula only works for heavy elements

    kk is the fraction of the Ba-star envelope that comes from the AGB
    """

    # Just apply the formula to each element
    new_model = np.log10((1 - kk) + kk * 10 ** model)

    # If ignoring first index
    if ignoreFirst:
        new_model[0] = model[0]

    return new_model

def get_one_gradient(model, data, dilution):
    """
    Get one gradient and sum of distances
    """

    # Useful values
    pow_mod = 10 ** model
    invLog = 2/np.log(10)
    x_k = (1 - dilution) + dilution * pow_mod
    diff = np.log10(x_k) - data

    # Gradient and distance
    grad = np.average((pow_mod - 1)/(data * x_k) * diff) * invLog
    sum_dist = np.sum(diff ** 2 / data)

    return sum_dist, grad

def get_distance(model, data):
    """
    Calculate a distance between model and data
    """

    # Chi square
    dist = np.average(np.abs((model - data) ** 2 / data))

    return dist
