import numpy as np

def apply_errors(arr, arr_err, nn):
    """
    Apply random errors drawn in a MC fashion
    """

    if nn > 0:
        new_arr = arr + np.random.random((nn, len(arr))) * 2 * arr_err - arr_err
    else:
        new_arr = arr + np.random.random((2, len(arr))) * 0

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

def calculate_dilution(data, model, processed_models = None):
    """
    Calculate best dilution for this model and data

    if processed_models is None, then model is taken to be a label
    otherwise model is taken as a model
    """

    if processed_models is not None:
        # Assign label
        label = model

        with open(processed_models, "r") as fread:
            # Read header
            fread.readline()

            # Find model
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
        # Apply dilution ignoring Fe/H
        dilut = apply_dilution(model, kk, ignoreFirst = True)

        # Check distance between data and diluted model
        dist = get_distance(dilut, data)

        # Save smallest distance
        if minDist is None or dist < minDist:
            minDist = dist
            minDil = kk

    return minDil, minDist

def get_distance(model, data):
    """
    Calculate a distance between model and data
    """

    # L square
    dist = np.mean((model - data) ** 2)

    return dist

def get_one_gradient(pow_mod, k_arr, x_k, log_x_k, data, k):
    """
    Get gradients
    """

    # Useful values
    invLog = 2/np.log(10)

    # First find the appropriate index
    indx = np.searchsorted(k_arr, k)

    # Now get diff
    diff = log_x_k[indx] - data

    # And the gradient
    grad = np.mean((pow_mod - 1)/x_k[indx] * diff, axis = 1) * invLog

    return grad

def find_k(model, data, tol = 1e-3):
    """
    Find the minimum dilution and sum of distances for this model
    """

    # Store the quantities that never change
    pow_mod = 10 ** model

    # Make the array x_k 10 times finer than the tolerance
    step = tol * 0.1
    k_arr = np.arange(0, 1 + step, step)
    x_k = np.array([(1 - k_arr) + k_arr * p for p in pow_mod]).T
    log_x_k = np.log10(x_k)

    # Start with the extremes
    k0 = 0
    k1 = 1
    km = 0.5

    while True:

        # Each of the gradients
        grad0 = get_one_gradient(pow_mod, k_arr, x_k, log_x_k, data, k0)
        gradm = get_one_gradient(pow_mod, k_arr, x_k, log_x_k, data, km)
        grad1 = get_one_gradient(pow_mod, k_arr, x_k, log_x_k, data, k1)

        # Get signs
        sign0 = np.sign(grad0)
        signm = np.sign(gradm)
        sign1 = np.sign(grad1)

        # Vectorized change
        k0 = np.abs(signm * (km + k0) + sign0 * (km - k0)) * 0.5
        k1 = np.abs(signm * (km + k1) + sign1 * (km - k1)) * 0.5

        # Get the middle point
        new_km = (k0 + k1) * 0.5

        # Check if converged
        dif = np.max(np.abs(new_km - km))
        if dif < tol:
            return new_km

        # Update
        km = new_km
