import numpy as np
import random
from datetime import datetime
random.seed(datetime.now())

class LSTM:
    def __init__(self, h_size, v_size):
        self.hidden_size = h_size
        self.vocab_size = v_size
        z_size = h_size + v_size

        # Weight matrix (forget gate)
        self.W_f = np.random.randn(h_size, z_size)

        # Bias for forget gate
        self.b_f = np.zeros((h_size, 1))

        # Weight matrix (input gate)
        self.W_i = np.random.randn(h_size, z_size)

        # Bias for input gate
        self.b_i = np.zeros((h_size, 1))

        # Weight matrix (candidate)
        self.W_g = np.random.randn(h_size, z_size)

        # Bias for candidate
        self.b_g = np.zeros((h_size, 1))

        # Weight matrix of the output gate
        self.W_o = np.random.randn(h_size, z_size)
        self.b_o = np.zeros((h_size, 1))

        # Weight matrix relating the hidden-state to the output
        self.W_v = np.random.randn(1, h_size)
        self.b_v = np.zeros((1, 1))

    def load(self, filename=None):
        if filename is None:
            name = f"LSTM_Parameters_H{self.vocab_size}_V{self.vocab_size}.txt"
        else:
            name = filename

        h_size = self.hidden_size
        v_size = self.vocab_size
        z_size = h_size + v_size

        paramfile = open(name)
        for i in range(3): paramfile.readline()
        
        load_matrix(self.W_f, h_size, z_size, paramfile)
        load_matrix(self.b_f, h_size, 1, paramfile)
        load_matrix(self.W_i, h_size, z_size, paramfile)
        load_matrix(self.b_i, h_size, 1, paramfile)
        load_matrix(self.W_g, h_size, z_size, paramfile)
        load_matrix(self.b_g, h_size, 1, paramfile)
        load_matrix(self.W_o, h_size, z_size, paramfile)
        load_matrix(self.b_o, h_size, 1, paramfile)
        load_matrix(self.W_v, 1, h_size, paramfile)
        load_matrix(self.b_v, 1, 1, paramfile)
        paramfile.close()

    def parameters(self):
        return (self.W_f, self.W_i, self.W_g, self.W_o, self.W_v, self.b_f, self.b_i, self.b_g, self.b_o, self.b_v)

    def forward(self, inputs, h_prev, C_prev):

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
            dh_prev = dz[:self.hidden_size, :]
            dC_prev = f[t] * dC

        grads = W_f_d, W_i_d, W_g_d, W_o_d, W_v_d, b_f_d, b_i_d, b_g_d, b_o_d, b_v_d
        grads = clip_gradient_norm(grads)

        return loss, grads

    def update(self, grads, lr=1e-3):  # Take a step
        params = self.parameters()
        for param, grad in zip(params, grads):
            param -= lr * grad

    def train(self, data, num_epochs, lr):
        for i in range(num_epochs):

            current_epoch_loss = 0
            spam_detected = 0
            spam_undetected = 0
            ham_detected = 0
            ham_undetected = 0

            for x, y in data:
                h = np.zeros((self.hidden_size, 1))
                c = np.zeros((self.hidden_size, 1))

                z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs = self.forward(x, h, c)
                result = outputs[len(outputs) - 1]

                loss, grads = self.backward(z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs, y)

                if y == 1:
                    if result >= 0.5:
                        ham_detected += 1
                    else:
                        ham_undetected += 1
                else:
                    if result < 0.5:
                        spam_detected += 1
                    else:
                        spam_undetected += 1

                self.update(grads, lr)

                current_epoch_loss += loss
            print(f"Epoch {i + 1}:")
            print(f"Training loss = {current_epoch_loss / len(data)},", end='')
            print(f" prediction accuracy = {100 * (ham_detected + spam_detected) / len(data)}%")
            print("{:<38}{:>4}, ".format("Correctly classified regular messages: ", ham_detected), end='')
            print("{:<32}{:>4}".format(" misclassified regular messages: ",ham_undetected))
            print("{:<38}{:>4}, ".format("Correctly classified spam messages: ", spam_detected), end='')
            print("{:<32}{:>4}".format(" misclassified spam messages: ", spam_undetected))
            print("*****************************************************")
            current_epoch_loss = 0
            spam_detected = 0
            spam_undetected = 0
            ham_detected = 0
            ham_undetected = 0

            if ((i + 1) % 20 == 0):
                W_f, W_i, W_g, W_o, W_v, b_f, b_i, b_g, b_o, b_v = self.parameters()
                file = open(f"LSTM_Parameters_H{self.vocab_size}_V{self.vocab_size}.txt", "w")
                file.write(" ************* LSTM Model Parameters ************* \n")

                file.write("\nW_f:\n")
                file.write(matrix_to_string(W_f))
                file.write("\nb_f matrix:\n")
                file.write(matrix_to_string(b_f))

                file.write("\nW_i matrix:\n")
                file.write(matrix_to_string(W_i))
                file.write("\nb_i matrix:\n")
                file.write(matrix_to_string(b_i))

                file.write("\nW_g matrix:\n")
                file.write(matrix_to_string(W_g))
                file.write("\nb_g matrix:\n")
                file.write(matrix_to_string(b_g))

                file.write("\nW_o matrix:\n")
                file.write(matrix_to_string(W_o))
                file.write("\nb_o matrix:\n")
                file.write(matrix_to_string(b_o))
                
                file.write("\nW_v matrix:\n")
                file.write(matrix_to_string(W_v))
                file.write("\nb_v matrix:\n")
                file.write(matrix_to_string(b_v))
                file.close()

def matrix_to_string(mat):
    result = ""
    rows, cols = mat.shape

    for row in range(rows):
        for col in range(cols):
            result += str(mat[row, col])
            result += "\t"
        result += "\n"

    return result

def load_matrix(mat, rows, cols, file):
    mat = np.zeros((rows, cols))

    for row in range(rows):
        values = file.readline().strip().split("\t")
        for col in range(cols):
            mat[row, col] += float(values[col])

    for i in range(2): file.readline()

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


