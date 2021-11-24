import numpy as np
import minitorch
from minitorch import tensor
from minitorch.autodiff import FunctionBase
from minitorch.tensor_ops import TensorOps
from minitorch.tensor_functions import TensorFunctions

import csv

import random
from datetime import datetime
random.seed(datetime.now())

vocab_size = 30
stupid_vocab_size = 2
hidden_size = 2
lr = 0.001
num_epochs = 500
batch_size = 100

def char_to_index(c):
    # 26 letters + punctuation + numbers + space
    char_num = ord(c)

    if char_num > 64 and char_num < 91:
        return char_num - 65

    if char_num > 96 and char_num < 123:
        return char_num - 97

    if char_num > 47 and char_num < 58:
        return 26

    if char_num < 33:
        return 27

    if char_num > 32 and char_num < 128:
        return 28

    return 29

def encode_char(c):
    one_hot_encoded = np.zeros((vocab_size, 1))
    one_hot_encoded[char_to_index(c), 0] = 1
    return one_hot_encoded

def decode_char(encoded):
    for i in range(vocab_size):
        if encoded[0, i] == 1:
            break
            
    if i == 29:
        return '@'

    if i == 28:
        return ','

    if i == 27:
        return ' '
    
    if i == 26:
        return '5'

    return chr(65 + i)

def decode_string(inputs):
    str = ""
    for input in inputs:
        str += decode_char(input)

    return str

class RNN:
    def __init__(self, hidden_size, vocab_size):
        # Weight matrix (input to hidden state)
        self.U = np.random.randn(hidden_size, vocab_size)

        # Weight matrix (recurrent computation)
        self.V = np.random.randn(hidden_size, hidden_size)

        # Weight matrix (hidden state to output)
        self.W = np.random.randn(1, hidden_size)

        # Bias (hidden state)
        self.b_hidden = np.random.randn(hidden_size, 1)

        # Bias (output)
        self.b_out = np.random.randn(1, 1)

    def parameters(self):
        return (self.U, self.V, self.W, self.b_hidden, self.b_out)

    def forward(self, inputs):

        hidden_state = np.zeros((hidden_size, 1))

        u, v, w, b_h, b = self.parameters()
        outputs, hidden_states = [], []

        # For each element in input sequence
        for t in range(len(inputs)):
            hidden_state = (u @ inputs[t]) + (v @ hidden_state) + b_h

            # Normalize hidden state
            #hidden_state = hidden_state - hidden_state.mean()
            #hidden_state = hidden_state / (hidden_state.std() + 1e-5)
            hidden_state = tanh(hidden_state)
            
            out = sigmoid((w @ hidden_state) + b)
            
            outputs.append(out)
            hidden_states.append(hidden_state.copy())
            #print(f"Hidden state before: {hidden_state}")
            #hidden_state = normalize(hidden_state, 2)
            #print(f"Hidden state after: {hidden_state}")
            #hidden_state /= np.linalg.norm(hidden_state)
            #hidden_state = hidden_state * 2

        return outputs, hidden_states

    def backward(self, inputs, outputs, hidden_states, target):
        """
        Computes the backward pass of a vanilla RNN.
        Args:
        `inputs`: sequence of inputs to be processed
        `outputs`: sequence of outputs from the forward pass `hidden_states`: sequence of hidden_states from the forward pass `targets`: sequence of targets
        `params`: the parameters of the RNN
        """
        # First we unpack our parameters
        u, v, w, b_h, b = self.parameters()

        # Initialize gradients as zero
        d_U, d_V, d_W = np.zeros_like(u), np.zeros_like(v), np.zeros_like(w)
        d_b_hidden, d_b_out = np.zeros_like(b_h), np.zeros_like(b)
        # Keep track of hidden state derivative and loss
        d_h_next = np.zeros_like(hidden_states[0])
        result = outputs[len(outputs) - 1]

        for t in reversed(range(len(outputs))):

            # Loss function used
            #loss = (result * y) + (result - 1.0) * (y - 1.0)
            #if loss == 0:
                #loss += 1e-5
            #loss = -np.log(loss)

            # Backpropogate into output
            prob = (outputs[t] * target) + (outputs[t] - 1.0) * (target - 1.0)
            d_o = -((2 * target - 1) / prob + 1e-5)
            d_o = d_o * sigmoid(outputs[t], derivative=True)

            # Backpropagate into W
            d_W += (d_o @ hidden_states[t].T)
            d_b_out += d_o

            # Backpropagate into h
            d_h = (w.T @ d_o) + d_h_next

            # Backpropagate through non-linearity
            d_f = tanh(hidden_states[t], derivative=True) * d_h
            d_b_hidden += d_f

            # Backpropagate into U
            d_U += (d_f @ inputs[t].T)

            # Backpropagate into V
            d_V += (d_f @ hidden_states[t - 1].T)
            d_h_next = (v.T @ d_f)

        grads = d_U, d_V, d_W, d_b_hidden, d_b_out
        grads = clip_gradient_norm(grads)

        return grads

    def update(self, grads, lr=1e-3):  # Take a step
        params = self.parameters()
        for param, grad in zip(params, grads):
            param -= lr * grad


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)

def sigmoid(x, derivative=False):
    """
    Computes the element-wise sigmoid activation function for an array x.
    Args:
     `x`: the array where the function is applied
     `derivative`: if set to True will return the derivative instead of the forward pass
    """
    x_safe = x + 1e-5
    f = 1 / (1 + np.exp(-x_safe))

    if derivative:
        # Return the derivative of the function evaluated at x
        return f * (1 - f)
    else:
        # Return the forward pass of the function at x
        return f

def tanh(x, derivative=False):
    """
    Computes the element-wise tanh activation function for an array x.
    Args:
     `x`: the array where the function is applied
     `derivative`: if set to True will return the derivative instead of the forward pass
    """
    x_safe = x + 1e-12
    f = (np.exp(x_safe)-np.exp(-x_safe))/(np.exp(x_safe)+np.exp(-x_safe ))

    if derivative:
        # Return the derivative of the function evaluated at x
        return 1-f**2
    else:
        # Return the forward pass of the function at x
        return f

def clip_gradient_norm(grads, max_norm=0.25):
    """
    Clips gradients to have a maximum norm of `max_norm`.
    This is to prevent the exploding gradients problem.
    """
    # Set the maximum of the norm to be of type float
    max_norm = float(max_norm)
    total_norm = 0
    # Calculate the L2 norm squared for each gradient and add them
    # to the total norm
    for grad in grads:
        grad_norm = np.sum(np.power(grad, 2))
        total_norm += grad_norm

    total_norm = np.sqrt(total_norm)

    # Calculate clipping coeficient
    clip_coef = max_norm / (total_norm + 1e-5)

    # If the total norm is larger than the maximum allowable norm,
    # then clip the gradient
    if clip_coef < 1:
        for grad in grads:
            grad *= clip_coef

    return grads

stupid_training_set = []

for i in range(1000):
    entry = []
    word_len = random.randint(1, 10)

    for j in range(word_len):
        enc = np.zeros((stupid_vocab_size, 1))
        rand = 2 * (random.random() - 0.5)
        
        if rand >= 0:
            enc[1, 0] = 1
        else:
            enc[0, 0] = 1
        entry.append(enc)

    sum = 0

    for char in entry:
        if char[0, 0] == 1:
            sum -= 1
        else:
            sum += 1

    if sum >= 0:
        stupid_training_set.append((entry, 1))
    else:
        stupid_training_set.append((entry, 0))

training_set = []

"""
with open('spam.csv', encoding = "ISO-8859-1") as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        break

    i = 0
    for row in spamreader:
        entry = []
        for c in row[1]:
            entry.append(encode_char(c))

        if row[0] == "ham":
            training_set.append((entry, 1))
        else:
            training_set.append((entry, 0))

        i += 1

        if i == 1000:
            break
"""
training_loss = []

model = RNN(hidden_size, stupid_vocab_size)

for i in range(num_epochs):

    current_epoch_loss = 0
    num_correct = 0
    batch_loss = 0
    count = 0

    for x, y in stupid_training_set:
        #print("Target: {}".format(y))
        # Forward pass
        #outputs, hidden_states, label = forward_pass(inputs, hidden_state, params)
        outputs, hidden_states = model.forward(x)
        
        result = outputs[len(outputs) - 1]
        #print("Output: {}".format(result))

        if y == 1 and result >= 0.5:
            num_correct += 1
        if y == 0 and result < 0.5:
            num_correct += 1

        # Compute loss
        loss = (result * y) + (result - 1.0) * (y - 1.0)

        if loss == 0:
            loss += 1e-5

        loss = -np.log(loss)

        #print("That gives us a loss of {}".format(loss))
        #print("")
        
        grads = model.backward(x, outputs, hidden_states, y)

        model.update(grads, lr)

        current_epoch_loss += loss
        batch_loss += loss
        count += 1

        """if count == batch_size:
            print("##########################################")
            print(f"Average loss over past {count} inputs: {batch_loss / count}")
            print("##########################################")
            print("")
            count = 0
            batch_loss = 0
            """

    print(f"Epoch {i}, training loss: {current_epoch_loss/len(stupid_training_set)}")
    print(f"Number of hits for this epoch = {num_correct}")
    print("****************************************")
    num_correct = 0
    u, v, w, b_h, b = model.parameters()

    if ((i+1) % 10 == 0):
        file = open("spam_filter_parameters.txt", "w")
        file.write(" ************* Parameters For Spam Filter Model ************* ")
        file.write("\nStopped at epoch {}".format(i+1))
        file.write("\nU matrix (input to hidden:")
        file.write(str(u))
        file.write("\nV matrix (hidden to hidden):")
        file.write(str(v))
        file.write("\nW matrix (hidden to output:")
        file.write(str(w))
        file.write("\nHidden state bias:")
        file.write(str(b_h))
        file.write("\nOutput bias:")
        file.write(str(b))
        file.write("")
        file.close()

        """
        print(" ************* Parameters For Spam Filter Model ************* ")
        print("\nStopped at epoch {}".format(i+1))
        print("\nU matrix (input to hidden:")
        print(str(u))
        print("\nV matrix (hidden to hidden):")
        print(str(v))
        print("\nW matrix (hidden to output:")
        print(str(w))
        print("\nHidden state bias:")
        print(str(b_h))
        print("\nOutput bias:")
        print(str(b))
        print("")
        """

