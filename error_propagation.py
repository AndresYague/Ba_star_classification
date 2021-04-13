import numpy as np
import matplotlib.pyplot as plt

class ErrorClass(object):
    """
    This class uses the error sensitivity tables from deCastro2016
    to calculate how the errors depend on each other
    """

    def __init__(self, error_tables = None, temperature_table = None):
        """
        Initialize variables and read tables
        """

        self.error_tables = error_tables
        self.temperature_table = temperature_table
        self.groups = None
        self.temperatures = None
        self.elements = None
        self.header = ["temp", "logg", "xi", "feh", "w"]

        # Load the tables automatically
        self.load_tables()

    def _transfom_numbers(self, char):
        """
        Transform numbers consistently
        """

        try:
            return float(char)
        except ValueError:
            return "-"
        except:
            raise

    def load_tables(self):
        """
        Load the error and temperature tables

        The error tables are loaded into the self.groups list as
        a dictionary with the temperature, the differences and
        each line errors

        The temperature tables are loaded into self.temperatures
        """

        if self.error_tables is not None:
            self.groups = []
            self.elements = set()
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
                            numbers = map(self._transfom_numbers, lnlst[2:])
                            self.groups[-1]["diff"] = list(numbers)

                        # Store the line name and the differences
                        # Store element names
                        else:
                            numbers = map(self._transfom_numbers, lnlst[1:])
                            name = lnlst[0]

                            self.groups[-1][name] = list(numbers)
                            self.elements.add(name)

                            # Give value of 0.1 to w when not defined
                            if self.groups[-1][name][4] == "-":
                                self.groups[-1][name][4] = 0.1

        if self.temperature_table is not None:
            self.temperatures = {}
            with open(self.temperature_table) as fread:

                # Skip header
                next(fread)

                # Now read
                for line in fread:
                    lnlst = line.split()
                    name = lnlst[0]
                    temperature = float(lnlst[1])
                    self.temperatures[name] = temperature

    def get_derivative(self, element, value, measure = "temp"):
        """
        Give interpolated derivative of desired measure
        """

        # Get the index
        try:
            index = self.header.index(measure)
        except ValueError:
            print("Accepted measures: ")
            print(self.header)
            raise
        except:
            raise

        # Get derivatives
        m_change = np.array([x[element][index] for x in self.groups])
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

    def calculate_errors(self, elements, elements_range, nn):
        """
        Calculate errors for elements. Those that appear in self.elements
        should be changed with the physical values
        """

        return NotImplementedError

    def plot_changes(self, measure = "temp", type_ = "derivative"):
        """
        Plot how the abundances depend on the index choosen
        """

        # Create the x_array
        all_x = [x[measure] for x in self.groups]
        all_x.sort()
        all_x = np.array(all_x)

        dx = (all_x[-1] - all_x[0]) * 0.1
        x_array = np.arange(all_x[0], all_x[-1] + dx, dx)

        # Now for every element plot the derivatives
        for element in self.elements:
            derivs = self.get_derivative(element, x_array, measure = measure)

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

    errors.plot_changes(measure = "xi", type_ = "derivative")
