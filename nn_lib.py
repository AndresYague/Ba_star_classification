import numpy as np

class NetworkObject(object):
    '''Class to create and manipulate neural networks
    the input is:

    -inpt: number of input neurons
    -hidden: list of length equal to hidden layers and each element
             is the number of neurons in each layer
    -output: number of output neurons
    -lbda: lambda parameter for regularization
    -fileName: if given, it will load appropriate network from file
    '''

    def __init__(self, inpt = 1, hidden = [], outpt = 1, lbda = 0.1,
                 fileName = None):

        # Load from filename if present
        self.fileName = fileName
        if fileName is not None:
            inpt, hidden, outpt = self.__parse_fileName(fileName)

        # Define sizes of layers
        self.inpt_num = inpt
        self.hidden_num = hidden
        self.outpt_num = outpt
        self.lbda = lbda

        # Now define theta and big deltas arrays
        self.theta_arrs = []
        self.big_deltas = []
        hid_len = len(self.hidden_num)

        # First theta
        num1 = self.inpt_num + 1
        if hid_len > 0:
            num2 = self.hidden_num[0]
            hid_len -= 1
        else:
            num2 = self.outpt_num

        sizRand = np.sqrt(6)/np.sqrt(num1 + num2)
        self.theta_arrs.append((np.random.random((num1, num2)) - 0.5) * sizRand)
        self.big_deltas.append(np.zeros((num1, num2)))

        # Subsequent
        ii = 1
        while hid_len > 0:
            num1 = num2 + 1
            num2 = self.hidden_num[ii]
            hid_len -= 1

            sizRand = np.sqrt(6)/np.sqrt(num1 + num2)
            self.theta_arrs.append((np.random.random((num1, num2)) - 0.5) * sizRand)
            self.big_deltas.append(np.zeros((num1, num2)))

        # Last one
        num1 = num2 + 1
        num2 = self.outpt_num

        sizRand = np.sqrt(6)/np.sqrt(num1 + num2)
        self.theta_arrs.append((np.random.random((num1, num2)) - 0.5) * sizRand)
        self.big_deltas.append(np.zeros((num1, num2)))

        if self.fileName is not None:
            self.__load_network()

    def __sigmoid(self, val):
        '''Calculate sigmoid function of val'''

        val = np.minimum(val, 100)
        val = np.maximum(val, -100)

        return 1/(1 + np.exp(-val))

    def __sigmoid_derv(self, val):
        '''Calculate sigmoid function derivative of val'''

        return self.__sigmoid(val) * (1 - self.__sigmoid(val))

    def get_cost(self, inpts, label_indices):
        '''Calculate cost function'''

        # Get output
        outputs = self.propagate(inpts)

        # Define label array
        # and add ones for each index in label_indices
        lab_arr = np.zeros(np.shape(outputs))
        for ii in range(len(lab_arr)):
            lab_arr[ii][label_indices[ii]] = 1

        # Now add all the costs
        cost = -sum(sum(lab_arr * np.log(outputs) +
                        (1 - lab_arr) * np.log(1 - outputs)))

        # Regularize
        for theta in self.theta_arrs:
            cost += sum(sum(theta[1:]**2)) * self.lbda * 0.5

        # Finally, divide by m
        cost /= len(inpts)

        return cost

    def calc_gradient(self, inpts, indices, batch_siz):
        '''Get gradients'''

        mm = batch_siz

        for ii in range(len(self.big_deltas)):
            self.big_deltas[ii] *= 0

        # Propagate forwards
        activations, activations_no_bias = self.propagate(inpts, grad = True)

        # Now backwards
        jj = -1
        delt_fin = activations_no_bias[jj]
        for ii in range(mm):
            delt_fin[ii][indices[ii]] -= 1

        # Save the deltas
        deltas = [delt_fin]

        ii = len(self.theta_arrs)
        while ii > 0:
            jj -= 1; ii -= 1

            # First add to big_deltas
            self.big_deltas[ii] += np.matmul(activations[jj].T, deltas[-1])

            # Now calculate new delta and add
            new_delt = np.matmul(deltas[-1], self.theta_arrs[ii][1:].T)
            new_delt *= activations_no_bias[jj] * (1 - activations_no_bias[jj])
            deltas.append(new_delt)

        # Add regularization and divide
        for ii in range(len(self.big_deltas)):
            self.big_deltas[ii][1:] += self.lbda * self.theta_arrs[ii][1:]
            self.big_deltas[ii] /= mm

    def train(self, train_inpts, label_indices, batch_siz = 10, cv_in = None,
              cv_lab = None, alpha = 0.1, verbose = False,
              tol = 1e-4, low_cost = 0.3, cycle_cost = 200):
        '''Train network with this example'''

        # Transform user input into np arrays if possible
        train_inpts = np.array(train_inpts)
        label_indices = np.array(label_indices)
        if cv_in is not None:
            cv_in = np.array(cv_in)
        if cv_lab is not None:
            cv_lab = np.array(cv_lab)

        ii = 0
        prevCost = None
        minCost = None
        cost_cv = None
        minCost_cv = None
        while True:
            # Define start and end indices
            init = ii * batch_siz
            end = min(init + batch_siz, len(train_inpts))

            # This specific batch size
            this_batch = end - init

            if ii % cycle_cost == 0:
                cost = self.get_cost(train_inpts, label_indices)

                # If use CV set, calculate cost
                if cv_in is not None and cv_lab is not None:
                    # Divide in its batch
                    init_cv = (ii % len(cv_lab)) * batch_siz
                    end_cv = min(init_cv + batch_siz, len(cv_lab))
                    init_cv = max(end_cv - batch_siz, 0)

                    cost_cv = self.get_cost(cv_in, cv_lab)

                # Register minimum cost
                if minCost is None or cost < minCost:
                    minCost = cost
                if minCost_cv is None or cost_cv < minCost_cv:
                    minCost_cv = cost_cv

                if verbose:
                    # Write the current cost
                    s = "The cost is {:4f}".format(cost)

                    # If use CV set, calculate cost
                    if cost_cv is not None:

                        s += " the cv cost is {:.4f}".format(cost_cv)

                    # Print minimum as well
                    s += " the minimum cost so far is {:.4f}".format(minCost)
                    if minCost_cv is not None:
                        s += " the minimum cv cost so far is {:.4f}".format(minCost_cv)

                    print(s)

                if prevCost is not None:
                    diff = abs(prevCost - cost)/cost
                    if diff < tol:
                        return cost

                prevCost = cost
                if cost < low_cost:
                    if cost_cv is not None:
                        if cost_cv < low_cost:
                            return cost
                    else:
                        return cost

            # Put the gradient calculation inside of a try
            # so user can cancel run and have a result
            try:
                self.calc_gradient(train_inpts[init:end],
                                   label_indices[init:end],
                                   batch_siz = this_batch)
            except KeyboardInterrupt:
                cost = self.get_cost(train_inpts, label_indices)
                return cost
            except:
                raise

            # Gradient descent
            for jj in range(len(self.big_deltas)):
                self.theta_arrs[jj] -= self.big_deltas[jj] * alpha

            ii += 1
            if end == len(train_inpts):
                ii = 0
                if verbose:
                    print("\n--> Starting again\n")

    def propagate_indx_conf(self, inpt_given):
        '''Give back the index and confidence after propagation'''

        output = self.propagate(inpt_given)
        idxMax = np.argmax(output, axis = 1)

        # Calculate the confidence
        sum_outputs = np.sum(output, axis = 1)
        conf = []
        for ii in range(len(idxMax)):
            conf.append(output[ii][idxMax[ii]]/sum_outputs[ii])

        return idxMax, conf

    def propagate(self, inpt_given, grad = False):
        '''Propagate network with given input and return output'''

        if len(np.shape(inpt_given)) == 1:
            inpt_given = np.reshape(inpt_given, (1, len(inpt_given)))

        # To add the one
        one = np.ones((len(inpt_given), 1))

        # Reshape so all are proper vectors
        no_bias = inpt_given
        curr_layer = np.append(one, no_bias, axis = 1)

        # If grad, add this one
        if grad:
            activations_no_bias = [no_bias]
            activations = [curr_layer]

        # Now propagate and add to activations
        for theta in self.theta_arrs:
            next_layer = self.__sigmoid(np.matmul(curr_layer, theta))
            curr_layer = np.append(one, next_layer, axis = 1)

            if grad:
                activations_no_bias.append(next_layer)
                activations.append(curr_layer)

        if grad:
            return activations, activations_no_bias
        else:
            return next_layer

    def get_thetas(self):
        '''Return the thetas'''

        return self.theta_arrs

    def set_thetas(self, thetas):
        '''Set the thetas to the given value'''

        self.theta_arrs = thetas

    def get_gradient(self):
        '''Return the gradient'''

        return self.big_deltas

    def __get_fileName(self, cost):
        '''Create a consistent filename for the network'''

        self.fileName = "saved_nn_{}_".format(self.inpt_num)
        for hid in self.hidden_num:
            self.fileName += "{}_".format(hid)
        self.fileName = self.fileName[:-1] + "_{}".format(self.outpt_num)
        self.fileName += "_cost_{:.3f}.npy".format(cost)

    def save_network(self, cost = 0):
        '''Save this network in file'''

        if self.fileName is None:
            self.__get_fileName(cost)

        # Trained, save thetas
        thetas = self.get_thetas()
        with open(self.fileName, "wb") as fwrite:
            for theta in thetas:
                np.save(fwrite, theta)

    def __load_network(self):
        '''Load network from file'''

        thetas = []
        nLayers = 2 + len(self.hidden_num)
        with open(self.fileName, "rb") as fread:
            for ii in range(nLayers - 1):
                thetas.append(np.load(fread))

        self.set_thetas(thetas)

    def __parse_fileName(self, fileName):
        '''Parse fileName to get parameters'''

        break_fileName = fileName.split("_")

        # Input and output
        inpt = int(break_fileName[2])
        outpt = int(break_fileName[-3])

        # Search for hidden layers
        hidden = list(map(lambda x: int(x), break_fileName[3:-3]))

        return inpt, hidden, outpt
