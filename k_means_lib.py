import numpy as np

class K_means(object):
    """
    Class for dealing with k-means algorithm
    """

    def __init__(self, data):
        """
        Constructor

        data should be a numpy array with all the data
        """

        # Store input
        self.data = np.array(data)
        self.n_dim = data.shape[1]

        # Initialize other attributes
        self.min_measure = None
        self.measures = []
        self.min_dictionary = None
        self.last_dictionary = None

    def do_k_means(self, n_k = 1, tol = 1e-1, attempts = 1):
        """
        Do the k-means with n_k number of groups and tolerance tol

        Take the best of attempts number of tries
        """

        # Restart variables
        self.min_measure = None
        self.measures = []

        # Do each attempt
        for attempt in range(attempts):

            # For each attempt get the measurement
            self._one_k_mean(n_k = n_k, tol = tol)
            self.measures.append(self.last_measure)

            # Get the dictionary if it's better
            if self.min_measure is None or self.last_measure < self.min_measure:
                self.min_measure = self.last_measure
                self.min_dictionary = self.last_dictionary

    def _one_k_mean(self, n_k, tol):
        """
        Do one k_mean algorithm

        Store current dictionary
        """

        # Get the limits for our k
        max_d = np.max(self.data, axis = 0)
        min_d = np.min(self.data, axis = 0)

        # Create the initial random k
        arr_k = []
        for k in range(n_k):
            this_k = np.random.random(self.n_dim) * (max_d - min_d) + min_d
            arr_k.append(this_k)

        arr_k = np.array(arr_k)

        # Now do the k-means
        move = tol + 1
        while move > tol:

            # Initialize list_of_groups
            list_of_groups = []
            for k in range(n_k):
                list_of_groups.append([])

            # First, classify the points

            # Take the distances of every datapoint to a k
            all_dists = []
            for k in arr_k:
                distances = np.sum((self.data - k) ** 2, axis = 1)
                all_dists.append(distances)

            # Transpose, to have the distances of datapoints to ks
            all_dists = np.transpose(all_dists)

            # Get the indices
            all_indices = np.argmin(all_dists, axis = 1)

            # Put each datapoint in its group
            for ii, index in enumerate(all_indices):
                list_of_groups[index].append(self.data[ii])

            # Now re-calculate arr_k
            new_arr_k = []
            for ii in range(len(list_of_groups)):

                # Get this group
                group = np.array(list_of_groups[ii])

                # Don't change if no elements here
                if len(group) == 0:
                    new_arr_k.append(arr_k[ii])
                    continue

                new_arr_k.append(np.mean(group, axis = 0))

            # Make into np.array
            new_arr_k = np.array(new_arr_k)

            # Calculate how much movement there was
            move = np.sum(np.abs((new_arr_k - arr_k) / arr_k))

            # And store new array, but drop the empty groups
            arr_k = []
            for group in new_arr_k:
                if len(group) > 0:
                    arr_k.append(group)
            arr_k = np.array(arr_k)

        # Create dictionary
        di = {}
        for k, group in zip(arr_k, list_of_groups):
            if len(group) > 0:
                di[tuple(k)] = np.array(group)
        self.last_dictionary = di

        # Now that it converged, get last measure
        self._get_measure()

    def _get_measure(self):
        """
        Give back how good this k-means is
        """

        measure = 0
        di = self.last_dictionary
        for key in di:
            measure += np.sum((np.array(key) - di[key]) ** 2)

        self.last_measure = measure

    def get_measurements(self):
        return self.measures

    def get_min_measurements(self):
        return self.min_measure

    def get_min_dictionary(self):
        return self.min_dictionary
