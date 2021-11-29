import numpy as np
import matplotlib.pyplot as plt
import os
import sys

class ErrorClass(object):
    """
    This class uses the error sensitivity tables from deCastro2016
    to calculate how the errors depend on each other
    """

    def __init__(self, error_tables = None, temperature_table = None,
                 element_set = None, verbose = 0):
        """
        Initialize variables and read tables
        """

        # Files
        self.error_tables = error_tables
        self.temperature_table = temperature_table
        self.element_set = element_set

        # Variables
        self.element_names = None
        self.groups = None
        self.average_group = None
        self.temperatures = None
        self.element_lines = None
        self.verbose = verbose

        # Header
        self.header = ["temp", "logg", "xi", "feh", "w"]

        # Load the tables automatically
        if self.element_set is not None:
            self.load_element_set()
        if self.error_tables is not None:
            self.load_error_tables()
        if self.temperature_table is not None:
            self.load_temperature_table()

    def _transform_numbers(self, char):
        """
        Transform numbers consistently
        """

        try:
            return float(char)
        except ValueError:
            return "-"
        except:
            raise

    def load_element_set(self):
        """
        Load the element set to use in this object
        It will be loaded into self.element_names
        """

        if self.element_set is None:
            s = f"{self.__str__()}.element_set is not specified"
            raise Exception(s)

        if not os.path.isfile(self.element_set):
            print(f"File {self.element_set} does not exist", file=sys.stderr)
            print(f"Element set not loaded", file=sys.stderr)
            return

        with open(self.element_set) as fread:
            for line in fread:
                lnlst = line.split()

                # Get first uncommented, non-empty line
                if len(lnlst) > 0 and lnlst[0][0] != "#":
                    this_set = lnlst
                    break

        # Remove "Fe/H" if present
        try:
            this_set.remove("Fe/H")
        except ValueError:
            pass
        except:
            raise

        # Remove the "/Fe" part of the name
        self.element_names = []
        for name in this_set:
            self.element_names.append(name.split("/")[0])

    def load_error_tables(self):
        """
        Load the error tables

        The error tables are loaded into the self.groups list as
        a dictionary with the temperature, the differences and
        each line errors
        """

        if self.error_tables is None:
            s = f"{self.__str__()}.error_tables is not specified"
            raise Exception(s)

        if not os.path.isfile(self.error_tables):
            print(f"File {self.error_tables} does not exist", file=sys.stderr)
            print(f"Error tables not loaded", file=sys.stderr)
            return

        self.groups = []
        self.average_group = {}
        self.element_lines = set()
        with open(self.error_tables) as fread:
            for line in fread:
                lnlst = line.split()

                if len(lnlst) > 0:

                    # Store temperature in kelvin
                    if "Temp" in line:
                        self.groups.append({})
                        self.groups[-1]["temp"] = float(lnlst[-2])

                    # Store the other values
                    elif "Logg" in line:
                        self.groups[-1]["logg"] = float(lnlst[-1])
                    elif "FeH" in line:
                        self.groups[-1]["feh"] = float(lnlst[-1])
                    elif "Xi" in line:
                        self.groups[-1]["xi"] = float(lnlst[-1])

                    # Store differences in the measurements
                    elif "Diff" in line:
                        numbers = list(map(self._transform_numbers, lnlst[2:]))
                        self.groups[-1]["diff"] = numbers

                        # Add to average group
                        self._add_to_average("diff", numbers)

                    # Store the line name and the differences
                    # Store element names
                    else:
                        numbers = list(map(self._transform_numbers, lnlst[1:]))

                        # Give value of 0.1 to w when not defined
                        if numbers[4] == "-":
                            numbers[4] = 0.1

                        norm = np.sqrt(np.sum(np.array(numbers[:-2])**2))
                        name = lnlst[0].split("I")[0]

                        self.groups[-1][name] = (numbers, norm)
                        self.element_lines.add(name)

                        # Add to average group
                        self._add_to_average(name, numbers)

        # Do average
        nn = len(self.groups)
        for key in self.average_group:
            for ii, val in enumerate(self.average_group[key]):

                # Skip undefined values
                if val == "-":
                    continue

                self.average_group[key][ii] /= nn

            # Nothing else to do
            if key == "diff":
                continue

            # If this key is not "diff", add the norm
            numbers = self.average_group[key]
            norm = np.sqrt(np.sum(np.array(numbers[:-2])**2))
            self.average_group[key] = (numbers, norm)

    def _add_to_average(self, key, new):
        """
        Add new list to key value of self.average_group
        """

        # If key is already there, add
        if key in self.average_group:
            next_ = self._add_lists(self.average_group[key], new)
            self.average_group[key] = next_

        # Otherwise, set new one
        else:
            self.average_group[key] = new

    def _add_lists(self, current, new):
        """
        Piecewise sum of mixed lists. Returns current + new
        """

        new_values = []
        for ii in range(len(current)):

            # If undefined value, return undefined value
            if current[ii] == "-" or new[ii] == "-":
                new_values.append("-")

            # Else, just add
            else:
                new_values.append(current[ii] + new[ii])

        return new_values

    def load_temperature_table(self):
        """
        Load the temperature table

        The temperature table is loaded into self.temperatures
        """

        if self.temperature_table is None:
            s = f"{self}.temperature_table is not specified"
            raise Exception(s)

        if not os.path.isfile(self.temperature_table):
            print(f"File {self.temperature_table} does not exist",
                  file=sys.stderr)
            print(f"Temperature table not loaded", file=sys.stderr)
            return

        self.temperatures = {}
        with open(self.temperature_table) as fread:

            # Skip header
            next(fread)

            # Now read
            for line in fread:
                lnlst = line.split()
                name = lnlst[0]
                temperature = float(lnlst[1])
                group = int(lnlst[2]) - 9
                self.temperatures[name] = (temperature, group)

    def calculate_errors(self, star_name, elements_range, nn, use_average=True):
        """
        Calculate errors for elements. Those that appear in self.element_lines
        should be changed with the physical values
        """

        # Create random errors
        len_ = len(elements_range)
        random_errors = np.random.normal(scale=elements_range, size=(nn, len_))

        # Return if tables not loaded
        if self.groups is None:
            return random_errors
        if self.temperatures is None:
            return random_errors
        if self.element_names is None:
            return random_errors

        # Return if the star is not on the table
        if star_name not in self.temperatures and not use_average:
            print("==============================", file=sys.stderr)
            print(f"{star_name} not in temperature table, skipping", file=sys.stderr)
            print("==============================", file=sys.stderr)
            self.plot_correlations(random_errors.T)
            return random_errors

        # Transpose to substitute
        random_errors = random_errors.T

        # Check which group to use:
        if use_average:
            group = self.average_group

        # Retrieve this star group
        else:
            temp, group = self.temperatures[star_name]
            group = self.groups[group]

        # Now specific errors
        diffs = np.array(group["diff"])
        random_changes = np.random.normal(scale=diffs,
                                          size=(nn, diffs.shape[0]))

        # To build a normal random variable x with std = sig from two other
        # normal random variables x1 and x2 with stds sig1 and sig2 and
        # derivatives of x with x1 and x2 of deriv1 and deriv2, one can
        # first calculate the distribution:
        #
        # x' = deriv1 * x1 + deriv2 * x2
        #
        # and then normalize with
        #
        # norm = sqrt((deriv1 * sig1)**2 + (deriv2*sig2)**2)
        #
        # such that x = x' * sig / norm

        # Put the Fe/H (this case has only one component and the derivative is 1)
        random_errors[0] = random_changes.T[3] / diffs[3] * elements_range[0]

        # For each element, sum the contributions of each error
        for ii, elem in enumerate(self.element_names):

            # If in the lines
            if elem in self.element_lines:

                # Apply changes
                diff_abund = np.array(group[elem][0][:-2])
                abund_changes = diff_abund / diffs * random_changes

                # Calculate total change and normalize
                total_change = np.sum(abund_changes, axis = 1) / group[elem][1]

                # And now multiply by the appropriate range
                total_change *= elements_range[ii + 1]

                # Finally substitute the random_errors
                # Knowing that we are not dealing with Fe/H
                random_errors[ii + 1] = total_change

        self.plot_correlations(random_errors)

        # Transpose back
        random_errors = random_errors.T

        return random_errors

    def plot_correlations(self, random_errors):
        """
        Plot the covariances between the errors
        """

        if self.verbose < 1:
            return

        # How many panels:
        names = ["Fe/H"] + [x + "/Fe" for x in self.element_names]
        nn = len(names)

        if self.verbose > 1:
            fig, axes = plt.subplots(nrows = nn, ncols = nn)

        for ii in range(len(names)):
            for jj in range(len(names)):

                if self.verbose > 1:
                    if ii == 0:
                        axes[ii][jj].set_title(names[jj])
                    if jj == 0:
                        axes[ii][jj].set_ylabel(names[ii])

                if jj < ii:
                    corr = np.corrcoef(random_errors[ii], random_errors[jj])
                    print(f"Correlation of {names[jj]} and {names[ii]} =",
                            end = " ")
                    print(f"{corr[0][1]:.2f}")

                if self.verbose > 1:
                    axes[ii][jj].plot(random_errors[ii], random_errors[jj], "o")

        if self.verbose > 1:
            plt.show()

    def _get_derivative(self, element, value, measure = "temp"):
        """
        Give interpolated derivative of desired measure
        """

        # Get the index
        try:
            index = self.header.index(measure)
        except ValueError:
            print("Accepted measures: ", file=sys.stderr)
            print(self.header, file=sys.stderr)
            raise
        except:
            raise

        # Get derivatives
        m_change = np.array([x[element][0][index] for x in self.groups])
        diffs = np.array([x["diff"][index] for x in self.groups])

        derivs = m_change / diffs

        # Find closest 2 values
        all_x = [x[measure] for x in self.groups]

        # Sort lists together
        zipped = list(zip(all_x, derivs))
        zipped.sort()

        # Unpack
        all_x, derivs = list(zip(*zipped))

        # Convert to numpy arrays
        all_x = np.array(all_x)
        derivs = np.array(derivs)

        # Interpolate
        return np.interp(value, all_x, derivs)

    def plot_changes(self, measure = "temp", type_ = "derivative"):
        """
        Plot how the abundances depend on the index choosen
        """

        # Create the x_array
        try:
            all_x = [x[measure] for x in self.groups]
        except KeyError:
            print("Accepted measures: ", file=sys.stderr)
            print(self.header[:-1], file=sys.stderr)
            raise
        except:
            raise

        all_x.sort()
        all_x = np.array(all_x)

        dx = (all_x[-1] - all_x[0]) * 0.1
        x_array = np.arange(all_x[0], all_x[-1] + dx, dx)

        # Now for every element plot the derivatives
        for element in self.element_lines:
            derivs = self._get_derivative(element, x_array, measure = measure)

            if type_ == "derivative":

                plt.plot(x_array, derivs, label = element, lw = 3)

            elif type_ == "change":

                y_arr = [0]
                dy_arr = derivs[0:-1] * (x_array[1:] - x_array[0:-1])
                for dy in dy_arr:
                    y_arr.append(y_arr[-1] + dy)

                plt.plot(x_array, y_arr, label = element, lw = 3)

        if type_ == "derivative":
            plt.ylabel("Derivative of abundance")
        elif type_ == "change":
            plt.ylabel("Cumulative change of abundance")

        plt.xlabel(measure)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    errors = ErrorClass(error_tables = "error_tables_ba.dat",
                        temperature_table = "bastars_temp.dat")

    errors.plot_changes(measure = "temp", type_ = "derivative")
