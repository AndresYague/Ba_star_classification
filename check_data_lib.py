import numpy as np
import os
import error_propagation

def modify_input(inputs):
    """
    Add features to the inputs so that they fit better the network
    """

    # Transpose for operation
    inputs = inputs.T

    # Old length
    old_len = inputs.shape[0]

    # Calculate new length
    new_len = (old_len - 1) * old_len // 2 + old_len

    # Initialize new input
    new_inputs = np.zeros((new_len, inputs.shape[1]))

    # Copy first part
    new_inputs[0:old_len] = inputs

    # Normalize
    new_inputs[1:] /= np.mean(np.abs(inputs[1:]), axis = 0)

    # Initialize values for loop
    init = old_len
    for ii in range(old_len):

        # Update slice
        slice_ = old_len - ii - 1

        # Substract
        new_inputs[init:init + slice_] = inputs[ii + 1:] - inputs[ii]

        # Update init
        init += slice_

    # Correct transposition
    new_inputs = new_inputs.T

    return new_inputs

def apply_errors(star_name, arr, arr_err, nn):
    """
    Apply random errors using ErrorClass.calculate_errors
    """

    if nn > 0:
        dir_path = "data_processing_and_plotting"
        errors = error_propagation.ErrorClass(
                error_tables = "error_tables_ba.dat",
                temperature_table = "bastars_temp.dat",
                element_set = os.path.join(dir_path, "element_set.dat"))

        error_diff = errors.calculate_errors(star_name, arr_err, nn)
        new_arr = arr + error_diff

    # If not applying errors
    else:
        new_arr = arr + np.random.random((1, len(arr))) * 0

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

def calculate_dilution(data, model, processed_models = None, lower = 0,
                       upper = 1):
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
            raise Exception(f"Label {label} not found")

        # Now transform into floats and np array
        model = np.array(list(map(lambda x: float(x), model)))

    # Dilute
    dk = 0.001
    dil_fact = np.arange(lower, upper + dk, dk)
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
    len_shape = len(data.shape)
    if len_shape == 2:
        dist = np.mean((model - data) ** 2, axis = 1)
    elif len_shape == 1:
        dist = np.mean((model - data) ** 2)
    else:
        raise NotImplementedError

    return dist

def get_one_gradient(k_arr, coef, log_x_k, data, k):
    """
    Get gradients
    """

    # First find the appropriate index
    indx = np.searchsorted(k_arr, k)

    # Now the whole gradient
    diff = (log_x_k[indx] - data) * coef[indx]

    # And the mean
    grad = np.mean(diff, axis = 1)

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

    # Get the x_k array
    x_k = np.array([(1 - k_arr) + k_arr * p for p in pow_mod]).T

    # And now the really useful arrays
    coef = (pow_mod - 1)/x_k * 2/np.log(10)
    log_x_k = np.log10(x_k)

    # Start with the extremes
    k0 = k_arr[0:1]
    k1 = k_arr[-1:]
    km = (k0 + k1) * 0.5

    first = True
    while True:

        # Each of the gradients
        gradm = get_one_gradient(k_arr, coef, log_x_k, data, km)
        if first:
            # If first time, we have to calculate everything
            grad0 = get_one_gradient(k_arr, coef, log_x_k, data, k0)
            grad1 = get_one_gradient(k_arr, coef, log_x_k, data, k1)

            first = False

        # Get signs
        sign0 = np.sign(grad0)
        signm = np.sign(gradm)
        sign1 = np.sign(grad1)

        # Get differences
        sum0 = np.abs(signm + sign0)
        dif0 = np.abs(signm - sign0)
        sum1 = np.abs(signm + sign1)
        dif1 = np.abs(signm - sign1)

        # Vectorized change
        k0 = (sum0 * km + dif0 * k0) * 0.5
        k1 = (sum1 * km + dif1 * k1) * 0.5

        # And change gradients too
        grad0 = (sum0 * gradm + dif0 * grad0) * 0.5
        grad1 = (sum1 * gradm + dif1 * grad1) * 0.5

        # Get the middle point
        new_km = (k0 + k1) * 0.5

        # Check if converged
        dif = np.max(np.abs(new_km - km))
        if dif < tol:
            return new_km

        # Update
        km = new_km

def get_list_networks(mod_dir):
    """
    Return list with each network directory
    """

    ii = 0
    networks_dir = []
    while True:

        # Calculate the subdirectory
        sub_dir = os.path.join(mod_dir, mod_dir + f"_{ii}")
        if not os.path.isdir(sub_dir):
            break

        networks_dir.append(sub_dir)
        ii += 1

    return networks_dir

def predict_with_networks(networks, inputs):
    """
    Predict with an ensemble of networks
    """

    all_predictions = []
    for network in networks:
        all_predictions.append(network.predict(inputs))

    return np.median(all_predictions, axis=0)
