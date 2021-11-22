import minitorch
import numpy as np
from minitorch import tensor
import csv
from minitorch.autodiff import FunctionBase
from minitorch.tensor_ops import TensorOps
from minitorch.tensor_functions import TensorFunctions
from minitorch import operators
from minitorch.tensor import Tensor
import random

vocab_size = 256
char_dict = {}

for i in range(vocab_size):
    #entry = Tensor.make([0], (1,), backend=TensorFunctions)
    entry = np.zeros((vocab_size,))
    entry[i] = 1
    char_dict[chr(i)] = entry

hidden_size = 20

def init_orthogonal(param):

    if param.ndim < 2:
        raise ValueError("Only parameters with 2 or more dimensions are supported")

    rows, cols = param.shape

    new_param = np.random.randn(rows, cols)

    if rows < cols:
        new_param = new_param.T

    # Compute QR factorization
    q, r = np.linalg.qr(new_param)

    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = np.diag(r, 0)
    ph = np.sign(d)
    q *= ph

    if rows < cols:
        q = q.T

    new_param = q
    
    return new_param

def init_rnn(hidden_size, vocab_size):
    """
    Initializes our recurrent neural network.
    Args:
     `hidden_size`: the dimensions of the hidden state
     `vocab_size`: the dimensions of our vocabulary
    """

    # Weight matrix (input to hidden state)
    U = np.ones((vocab_size, hidden_size))
    # Weight matrix (recurrent computation)
    V = np.zeros((hidden_size, hidden_size))

    # Weight matrix (hidden state to output)
    W = np.zeros((hidden_size, 1))

    # Bias (hidden state)
    b_hidden = np.zeros((hidden_size,))

    # Bias (output)
    b_out = np.zeros((1,))

    # Initialize weights
    U = init_orthogonal(U)
    V = init_orthogonal(V)
    W = init_orthogonal(W)

    # Return parameters as a tuple
    return U, V, W, b_hidden, b_out

params = init_rnn(hidden_size=hidden_size, vocab_size=vocab_size)

def sigmoid(x, derivative=False):
    """
    Computes the element-wise sigmoid activation function for an array x.
    Args:
     `x`: the array where the function is applied
     `derivative`: if set to True will return the derivative instead of the forward pass
    """
    x_safe = x + 1e-12
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

def softmax(x, derivative=False):
    """
    Computes the softmax for an array x.
    Args:
     `x`: the array where the function is applied
     `derivative`: if set to True will return the derivative instead of the forward pass
    """
    x_safe = x + 1e-12
    f = np.exp(x_safe) / np.sum(np.exp(x_safe))

    if derivative:
        # Return the derivative of the function evaluated at x
        pass # We will not need this one
    else:
        # Return the forward pass of the function at x
        return f

def forward_pass(inputs, hidden_state, params):
    """
    Computes the forward pass of a vanilla RNN.
    Args:
     `inputs`: sequence of inputs to be processed
     `hidden_state`: an already initialized hidden state
     `params`: the parameters of the RNN
    """
    # First we unpack our parameters
    U, V, W, b_hidden, b_out = params

    # Create a list to store outputs and hidden states
    hidden_states = []

    # For each element in input sequence
    for t in range(len(inputs)):
        # Compute new hidden state
        print("Current input:")
        print(inputs[t])
        hidden_state = tanh(np.dot(inputs[t], U) + np.dot(V, hidden_state) + b_hidden)
        print("Current hidden state:")
        print(hidden_state)
        # Compute output
        out = sigmoid(np.dot(hidden_state, W) + b_out)
        print("Current output:")
        print(out)
        # Save results and continue
        hidden_states.append(hidden_state.copy())
        print("")
        """print("Input shape:")
                print(inputs[t].shape)
                print("Hidden state shape:")
                print(hidden_state.shape)
                print("U shape:")
                print(U.shape)
                print("V shape:")
                print(V.shape)
                print("W shape:")
                print(W.shape)

                print("input x U shape:")
                print(np.dot(inputs[t], U).shape)

                print("V x hidden shape:")
                print(np.dot(V, hidden_state).shape)
                """
    return out, hidden_states

def clip_gradient_norm(grads, max_norm=0.25):
    """
    Clips gradients to have a maximum norm of `max_norm`.
    This is to prevent the exploding gradients problem.
    """
    # Set the maximum of the norm to be of type float max_norm = float(max_norm)
    total_norm = 0
    # Calculate the L2 norm squared for each gradient and add them to th
    e total norm
    for grad in grads:
        grad_norm = np.sum(np.power(grad, 2))
        total_norm += grad_norm
    total_norm = np.sqrt(total_norm)

    # Calculate clipping coeficient
    clip_coef = max_norm / (total_norm + 1e-6)

    # If the total norm is larger than the maximum allowable norm,
    # then clip the gradient
    if clip_coef < 1:
        for grad in grads:
            grad *= clip_coef return grads

def backward_pass(inputs, outputs, hidden_states, targets, params):
    """
    Computes the backward pass of a vanilla RNN.
    Args:
    `inputs`: sequence of inputs to be processed
    `outputs`: sequence of outputs from the forward pass `hidden_states`: sequence of hidden_states from the forward pass `targets`: sequence of targets
    `params`: the parameters of the RNN
    """
        # First we unpack our parameters
    U, V, W, b_hidden, b_out = params
        # Initialize gradients as zero
    d_U, d_V, d_W = np.zeros_like(U), np.zeros_like(V), np.zeros_like(W) d_b_hidden, d_b_out = np.zeros_like(b_hidden), np.zeros_like(b_out)
        # Keep track of hidden state derivative and loss
    d_h_next = np.zeros_like(hidden_states[0]) loss = 0


test_seq = []

with open('spam.csv', encoding = "ISO-8859-1") as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        break
    for row in spamreader:
        break

    for row in spamreader:
        for c in row[1]:
            test_seq.append(char_dict[c])
        break

# Get first sequence in training set
#test_input_sequence, test_target_sequence = training_set[0]
# One-hot encode input and target sequence
#test_input = one_hot_encode_sequence(test_input_sequence, vocab_size) test_target = one_hot_encode_sequence(test_target_sequence, vocab_size)
# Initialize hidden state as zeros

hidden_state = np.zeros((hidden_size,)) # Now let's try out our new function
output, hidden_states = forward_pass(test_seq, hidden_state, params)
#print('Input sequence:')
#print(test_seq)
print('\nPredicted sequence:')
print(output)
