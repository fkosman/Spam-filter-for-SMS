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
hidden_size = 10
lr = 0.005
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

        U, V, W, b_h, b = self.parameters()
        outputs, hidden_states = [], []

        # For each element in input sequence
        for t in range(len(inputs)):
            hidden_state = (U @ inputs[t]) + (V @ hidden_state) + b_h

            # Normalize hidden state
            #hidden_state = hidden_state - hidden_state.mean()
            #hidden_state = hidden_state / (hidden_state.std() + 1e-5)
            hidden_state = tanh(hidden_state)
            
            out = sigmoid((W @ hidden_state) + b)
            
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
        U, V, W, b_h, b = self.parameters()

        # Initialize gradients as zero
        d_U, d_V, d_W = np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)
        d_b_hidden, d_b_out = np.zeros_like(b_h), np.zeros_like(b)
        # Keep track of hidden state derivative and loss
        d_h_next = np.zeros_like(hidden_states[0])
        loss = 0

        for t in reversed(range(len(outputs))):

            # Add loss
            loss += np.absolute(outputs[t] - target)
            
            # Backpropogate into output
            d_o = (outputs[t] - target) / np.absolute(outputs[t] - target + 1e-5)
            
            d_o = d_o * sigmoid(outputs[t], derivative=True)

            # Backpropagate into W
            d_W += (d_o @ hidden_states[t].T)
            d_b_out += d_o

            # Backpropagate into h
            d_h = (W.T @ d_o) + d_h_next

            # Backpropagate through non-linearity
            d_f = tanh(hidden_states[t], derivative=True) * d_h
            d_b_hidden += d_f

            # Backpropagate into U
            d_U += (d_f @ inputs[t].T)

            # Backpropagate into V
            d_V += (d_f @ hidden_states[t - 1].T)
            d_h_next = (V.T @ d_f)

        grads = d_U, d_V, d_W, d_b_hidden, d_b_out
        grads = clip_gradient_norm(grads)

        return loss, grads

    def update(self, grads, lr=1e-3):  # Take a step
        params = self.parameters()
        for param, grad in zip(params, grads):
            param -= lr * grad

class LSTM:
    def __init__(self, hidden_size, vocab_size):
        z_size = hidden_size + vocab_size

        # Weight matrix (forget gate)
        self.W_f = np.random.randn(hidden_size, z_size)

        # Bias for forget gate
        self.b_f = np.zeros((hidden_size, 1))

        # Weight matrix (input gate)
        self.W_i = np.random.randn(hidden_size, z_size)

        # Bias for input gate
        self.b_i = np.zeros((hidden_size, 1))

        # Weight matrix (candidate)
        self.W_g = np.random.randn(hidden_size, z_size)

        # Bias for candidate
        self.b_g = np.zeros((hidden_size, 1))

        # Weight matrix of the output gate
        self.W_o = np.random.randn(hidden_size, z_size)
        self.b_o = np.zeros((hidden_size, 1))

        # Weight matrix relating the hidden-state to the output
        self.W_v = np.random.randn(1, hidden_size)
        self.b_v = np.zeros((1, 1))

    def parameters(self):
        return (self.W_f, self.W_i, self.W_g, self.W_o, self.W_v, self.b_f, self.b_i, self.b_g, self.b_o, self.b_v)

    def forward(self, inputs, h_prev, C_prev):

        assert h_prev.shape == (hidden_size, 1)
        assert C_prev.shape == (hidden_size, 1)

        W_f, W_i, W_g, W_o, W_v, b_f, b_i, b_g, b_o, b_v = self.parameters()

        # Save a list of computations for each of the components in the LSTM
        x_s, z_s, f_s, i_s, = [], [], [], []
        g_s, C_s, o_s, h_s = [], [], [], []
        v_s, output_s = [], []

        # Append the initial cell and hidden state to their respective lists
        h_s.append(h_prev)
        C_s.append(C_prev)

        for x in inputs:
            z = np.row_stack((h_prev, x))
            z_s.append(z)

            # Calculate forget gate
            f = sigmoid((W_f @ z) + b_f)
            f_s.append(f)

            # Calculate input gate
            i = sigmoid((W_i @ z) + b_i)
            i_s.append(i)

            # Calculate candidate
            g = tanh((W_g @ z) + b_g)
            g_s.append(g)

            # Calculate memory state
            C_prev = f * C_prev + i * g
            C_s.append(C_prev)

            # Calculate output gate
            o = sigmoid((W_o @ z) + b_o)
            o_s.append(o)

            # Calculate hidden state
            h_prev = o * tanh(C_prev)
            h_s.append(h_prev)

            # Calculate logits
            v = (W_v @ h_prev) + b_v
            v_s.append(v)

            # Calculate sigmoid activation
            output = sigmoid(v)
            output_s.append(output)
        
        return z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, output_s

    def backward(self, z, f, i, g, C, o, h, v, outputs, target):
        """
        Computes the backward pass of an LSTM.
        Args:
        """
        # First we unpack our parameters
        W_f, W_i, W_g, W_o, W_v, b_f, b_i, b_g, b_o, b_v = self.parameters()

        # Initialize gradients as zero
        W_f_d = np.zeros_like(W_f)
        b_f_d = np.zeros_like(b_f)
        W_i_d = np.zeros_like(W_i)
        b_i_d = np.zeros_like(b_i)
        W_g_d = np.zeros_like(W_g)
        b_g_d = np.zeros_like(b_g)
        W_o_d = np.zeros_like(W_o)
        b_o_d = np.zeros_like(b_o)
        W_v_d = np.zeros_like(W_v)
        b_v_d = np.zeros_like(b_v)

        # Set the next cell and hidden state equal to zero
        dh_next = np.zeros_like(h[0])
        dC_next = np.zeros_like(C[0])

        loss = 0

        for t in reversed(range(len(outputs))):

            # Add loss
            loss += np.absolute(outputs[t] - target)

            # Get the previous hidden cell state
            C_prev = C[t - 1]

            # Backpropogate into output
            dv = (outputs[t] - target) / np.absolute(outputs[t] - target + 1e-5)
            dv = dv * sigmoid(outputs[t], derivative=True)

            # Update the gradient of the relation of the hidden-state to the output gatents
            W_v_d += (dv @ h[t].T)
            b_v_d += dv

            # Compute the derivative of the hidden state and output gate
            dh = (W_v.T @ dv)
            dh += dh_next
            do = dh * tanh(C[t])
            do = sigmoid(o[t], derivative=True) * do

            # Update the gradients with respect to the output gate
            W_o_d += (do @ z[t].T)
            b_o_d += do

            # Compute the derivative of the cell state and candidate g
            dC = np.copy(dC_next)
            dC += dh * o[t] * tanh(tanh(C[t]), derivative=True)
            dg = dC * i[t]
            dg = tanh(g[t], derivative=True) * dg

            # Update the gradients with respect to the candidate
            W_g_d += (dg @ z[t].T)
            b_g_d += dg

            # Compute the derivative of the input gate and update its gradients
            di = dC * g[t]
            di = sigmoid(i[t], True) * di
            W_i_d += (di @ z[t].T)
            b_i_d += di

            # Compute the derivative of the forget gate and update its gradients
            df = dC * C_prev
            df = sigmoid(f[t]) * df
            W_f_d += (df @ z[t].T)
            b_f_d += df

            # Compute the derivative of the input and update the gradients of the
            # previous hidden and cell state
            dz = ((W_f.T @ df) + (W_i.T @ di) + (W_g.T @ dg) + (W_o.T @ do))
            dh_prev = dz[:hidden_size, :]
            dC_prev = f[t] * dC

        grads = W_f_d, W_i_d, W_g_d, W_o_d, W_v_d, b_f_d, b_i_d, b_g_d, b_o_d, b_v_d
        grads = clip_gradient_norm(grads)

        return loss, grads

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
model2 = LSTM(hidden_size, stupid_vocab_size)

for i in range(num_epochs):

    current_epoch_loss = 0
    num_correct = 0
    batch_loss = 0
    count = 0

    for x, y in stupid_training_set:
        #print("Target: {}".format(y))
        # Forward pass
        #outputs, hidden_states, label = forward_pass(inputs, hidden_state, params)

        #outputs, hidden_states = model.forward(x)
        h = np.zeros((hidden_size, 1))
        c = np.zeros((hidden_size, 1))

        z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs = model2.forward(x, h, c)
        result = outputs[len(outputs) - 1]
        # print("Output: {}".format(result))

        loss, grads = model2.backward(z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs, y)

        if y == 1 and result >= 0.5:
            num_correct += 1
        if y == 0 and result < 0.5:
            num_correct += 1

        #print("That gives us a loss of {}".format(loss))
        #print("")
        
        #loss, grads = model.backward(x, outputs, hidden_states, y)

        model2.update(grads, lr)

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

    if ((i+1) % 50 == 0):
        file = open("spam_filter_parameters.txt", "w")
        file.write(" ************* Parameters For Spam Filter Model ************* ")
        file.write("\nStopped at epoch {}".format(i+1))
        file.write("\nU matrix (input to hidden):")
        file.write(str(u))
        file.write("\nV matrix (hidden to hidden):")
        file.write(str(v))
        file.write("\nW matrix (hidden to output):")
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
        print("\nU matrix (input to hidden):")
        print(str(u))
        print("\nV matrix (hidden to hidden):")
        print(str(v))
        print("\nW matrix (hidden to output):")
        print(str(w))
        print("\nHidden state bias:")
        print(str(b_h))
        print("\nOutput bias:")
        print(str(b))
        print("")
        """

