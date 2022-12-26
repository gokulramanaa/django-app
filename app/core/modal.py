import numpy as np
import os
import random
import json
from app.core.utils import *
from django.conf import settings
import pickle

char_to_ix = {'\n': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26}
ix_to_char = {0: '\n', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z'}

class NameGenerator():
    def __init__(self) -> None:
        pass

    def clip(self, gradients, maxValue):
        '''
        self.clips the gradients' values between minimum and maximum.

        Arguments:
        gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
        maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue

        Returns: 
        gradients -- a dictionary with the self.clipped gradients.
        '''

        dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients[
            'dWax'], gradients['dWya'], gradients['db'], gradients['dby']

        # self.clip to mitigate exploding gradients, loop over [dWax, dWaa, dWya, db, dby]. (≈2 lines)
        for gradient in [dWax, dWaa, dWya, db, dby]:
            np.clip(gradient, -maxValue, maxValue, out=gradient)

        gradients = {"dWaa": dWaa, "dWax": dWax,
                    "dWya": dWya, "db": db, "dby": dby}

        return gradients


    # def setup(self):
    #     data = open(os.path.join(settings.DATA_DIR, 'girlname.txt'), 'r').read()
    #     data = data.lower()
    #     chars = list(set(data))
    #     data_size, vocab_size = len(data), len(chars)
    #     print('There are %d total characters and %d unique characters in your data.' % (
    #         data_size, vocab_size))

    #     char_to_ix = {ch: i for i, ch in enumerate(sorted(chars))}
    #     ix_to_char = {i: ch for i, ch in enumerate(sorted(chars))}
    #     return (data, char_to_ix, ix_to_char)


    def sample(self, parameters, seed):
        """
        self.sample a sequence of characters according to a sequence of probability distributions output of the RNN

        Arguments:
        parameters -- python dictionary containing the parameters Waa, Wax, Wya, by, and b. 
        char_to_ix -- python dictionary mapping each character to an index.
        seed -- used for grading purposes. Do not worry about it.

        Returns:
        indices -- a list of length n containing the indices of the self.sampled characters.
        """

        # Retrieve parameters and relevant shapes from "parameters" dictionary
        Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
        vocab_size = by.shape[0]
        n_a = Waa.shape[1]

        # Step 1: Create the one-hot vector x for the first character (initializing the sequence generation). (≈1 line)
        x = np.zeros((vocab_size, 1))
        # Step 1': Initialize a_prev as zeros (≈1 line)
        a_prev = np.zeros((n_a, 1))

        # Create an empty list of indices, this is the list which will contain the list of indices of the characters to generate (≈1 line)
        indices = []

        # Idx is a flag to detect a newline character, we initialize it to -1
        idx = -1

        # Loop over time-steps t. At each time-step, self.sample a character from a probability distribution and append
        # its index to "indices". We'll stop if we reach 50 characters (which should be very unlikely with a well
        # trained self.model), which helps debugging and prevents entering an infinite loop.
        counter = 0
        newline_character = char_to_ix['\n']

        while (idx != newline_character and counter != 50):

            # Step 2: Forward propagate x using the equations (1), (2) and (3)
            a = np.tanh(np.add(np.add(np.dot(Wax, x), np.dot(Waa, a_prev)), b))
            z = np.add(np.dot(Wya, a), by)
            y = softmax(z)

            # for grading purposes
            np.random.seed(counter+seed)

            # Step 3: self.sample the index of a character within the vocabulary from the probability distribution y
            idx = np.random.choice(list(range(vocab_size)), p=y.ravel())

            # Append the index to "indices"
            indices.append(idx)

            # Step 4: Overwrite the input character as the one corresponding to the self.sampled index.
            x = np.zeros((vocab_size, 1))
            x[idx] = 1

            # Update "a_prev" to be "a"
            a_prev = a

            # for grading purposes
            seed += 1
            counter += 1

        if (counter == 50):
            indices.append(char_to_ix['\n'])

        return indices


    def optimize(self, X, Y, a_prev, parameters, learning_rate=0.01):
        """
        Execute one step of the optimization to train the self.model.

        Arguments:
        X -- list of integers, where each integer is a number that maps to a character in the vocabulary.
        Y -- list of integers, exactly the same as X but shifted one index to the left.
        a_prev -- previous hidden state.
        parameters -- python dictionary containing:
                            Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                            Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                            Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                            b --  Bias, numpy array of shape (n_a, 1)
                            by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
        learning_rate -- learning rate for the self.model.

        Returns:
        loss -- value of the loss function (cross-entropy)
        gradients -- python dictionary containing:
                            dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                            dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                            dWya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)
                            db -- Gradients of bias vector, of shape (n_a, 1)
                            dby -- Gradients of output bias vector, of shape (n_y, 1)
        a[len(X)-1] -- the last hidden state, of shape (n_a, 1)
        """

        # Forward propagate through time (≈1 line)
        loss, cache = rnn_forward(X, Y, a_prev, parameters)

        # Backpropagate through time (≈1 line)
        gradients, a = rnn_backward(X, Y, parameters, cache)

        # self.clip your gradients between -5 (min) and 5 (max) (≈1 line)
        gradients = self.clip(gradients, 5)

        # Update parameters (≈1 line)
        parameters = update_parameters(parameters, gradients, learning_rate)

        return loss, gradients, a[len(X)-1]


    def model(self, num_iterations=50000, n_a=50, dino_names=7, vocab_size=27):
        """
        Trains the self.model and generates dinosaur names. 

        Arguments:
        data -- text corpus
        ix_to_char -- dictionary that maps the index to a character
        char_to_ix -- dictionary that maps a character to an index
        num_iterations -- number of iterations to train the self.model for
        n_a -- number of units of the RNN cell
        dino_names -- number of dinosaur names you want to self.sample at each iteration. 
        vocab_size -- number of unique characters found in the text, size of the vocabulary

        Returns:
        parameters -- learned parameters
        """

        # Retrieve n_x and n_y from vocab_size
        n_x, n_y = vocab_size, vocab_size

        # Initialize parameters
        parameters = initialize_parameters(n_a, n_x, n_y)

        # Initialize loss (this is required because we want to smooth our loss, don't worry about it)
        loss = get_initial_loss(vocab_size, dino_names)

        # Build list of all dinosaur names (training examples).
        with open(os.path.join(settings.DATA_DIR, 'girlname.txt')) as f:
            examples = f.readlines()
        examples = [x.lower().strip() for x in examples]
        # print(len(examples))
        # print(examples[:5])

        # Shuffle list of all dinosaur names
        np.random.seed(0)
        np.random.shuffle(examples)
        # print(examples[:5])

        # Initialize the hidden state of your LSTM
        a_prev = np.zeros((n_a, 1))
        res = []

        # Optimization loop
        for j in range(num_iterations):

            # Use the hint above to define one training example (X,Y) (≈ 2 lines)
            index = j % len(examples)
            # print(index)
            X = [None] + [char_to_ix[ch] for ch in examples[index]]
            Y = X[1:] + [char_to_ix["\n"]]
            # print(X)
            # print(Y)

            # Perform one optimization step: Forward-prop -> Backward-prop -> self.clip -> Update parameters
            # Choose a learning rate of 0.01
            curr_loss, gradients, a_prev = self.optimize(X, Y, a_prev, parameters)

            # Use a latency trick to keep the loss smooth. It happens here to accelerate the training.
            loss = smooth(loss, curr_loss)

            # Every 2000 Iteration, generate "n" characters thanks to self.sample() to check if the self.model is learning properly
            if j % 2000 == 0:

                print('Iteration: %d, Loss: %f' % (j, loss) + '\n')

                # The number of dinosaur names to print
                seed = 0
                for name in range(dino_names):

                    # self.sample indices and print them
                    self.sampled_indices = self.sample(parameters, seed)
                    # print_sample(self.sampled_indices, ix_to_char)
                    res.append(get_sample(self.sampled_indices, ix_to_char))

                    # To get the same result for grading purposed, increment the seed by one.
                    seed += 1

                print('\n')

        return parameters, res

    def get_names(self, num_result=10):
        with open("./modaldata.pkl", "rb") as outfile:
            parameters = pickle.load(outfile)

        if not parameters:
            return []
        res = []
        seed = random.randrange(100)
        for _ in range(int(num_result)):
            self.sampled_indices = self.sample(parameters, seed)
            res.append(get_sample(self.sampled_indices, ix_to_char))
            seed += 1
        return res

    def set_values(self,):
        parameters, res = self.model()
        print(type(parameters))
        with open("./modaldata.pkl", "rb") as infile:
            try:
                values = pickle.load(infile)
            except:
                values = {}
            values.update(parameters)

        print(values, type(values))
        with open("./modaldata.pkl", "wb") as outfile:
            pickle.dump(values, outfile)

