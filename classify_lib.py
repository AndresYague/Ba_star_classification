import numpy as np
import os
import error_propagation

class StarStat(object):
    """
    Class for all the stellar statistics
    """

    def __init__(self, star_name, values_arr, errors_arr, nn=1e5):
        """
        Initialize the StarStat instance
        """

        self.star_name = star_name
        self.values_arr = values_arr
        self.errors_arr = errors_arr
        self.nn = int(nn)
        self.chisq = None

    def _apply_errors(self):
        """
        Apply random errors using ErrorClass.calculate_errors
        """

        if self.nn > 0:
            dir_path = "data_processing_and_plotting"
            errors = error_propagation.ErrorClass(
                    error_tables = "error_tables_ba.dat",
                    temperature_table = "bastars_temp.dat",
                    element_set = os.path.join(dir_path, "element_set.dat"))

            error_diff = errors.calculate_errors(self.star_name,
                                                 self.errors_arr, self.nn)
            new_arr = self.values_arr + error_diff

        # If not applying errors
        else:
            new_arr = self.values_arr + np.random.random((1, len(arr))) * 0

        return new_arr

    def goodness_of_fit(self, model):
        """
        Calculate the goodness of fit for this model
        """

        # Get modified chi-2
        if self.chisq is None:
            mc_values = self._apply_errors()
            chisq = (mc_values - self.values_arr)**2/self.errors_arr

            # Sort it
            self.chisq = np.sort(np.sum(chisq, axis=1))

        # Modified chi-2 value for the model
        if len(model.shape) == 2:
            chisq_mod = np.sum((model - self.values_arr)**2/self.errors_arr, axis=1)
        else:
            chisq_mod = np.sum((model - self.values_arr)**2/self.errors_arr)

        # Search index
        indices = np.searchsorted(self.chisq, chisq_mod)

        # Probability of equal or better
        pVal = 1 - indices/self.nn

        return pVal

    def calculate_dilution(self, model, k_step=1e-3, max_dil=1.0):
        """
        Calculate the best dilution according to goodness of fit for this model
        """

        kk = 0
        pVal_and_k = []

        # Dilute from 0 to max_dil
        while kk < max_dil:
            # Diluted model
            dil_model = apply_dilution(model, kk, ignoreFirst=True)

            # Goodness of fit
            pVal = self.goodness_of_fit(dil_model)

            # Add this goodness of fit and dilution
            pVal_and_k.append((pVal, kk))

            kk += k_step

        # Sort by pVal
        pVal_and_k.sort(reverse=True)

        # Return best
        return pVal_and_k[0]

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

def apply_dilution(model, kk, ignoreFirst=False):
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

    # Ensemble predictions
    all_predictions = []
    for network in networks:
        all_predictions.append(network.predict(inputs))

    # Predict with median, also return all predictions
    return np.median(all_predictions, axis=0), all_predictions
