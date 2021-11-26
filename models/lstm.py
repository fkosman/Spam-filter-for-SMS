import numpy as np
import os.path
from models.header import *

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
            name = f"saved/LSTM_Parameters_H{self.hidden_size}_V{self.vocab_size}.params"
        else:
            name = "saved/" + filename + ".params"

        if not os.path.isfile(name):
            print("No parameter file was found.")
            return
        
        with open(name) as paramfile:
            paramfile.readline()
            for param in self.parameters(): load_matrix(param, paramfile)

    def save(self, filename=None):
        if filename is None:
            name = f"saved/LSTM_Parameters_H{self.hidden_size}_V{self.vocab_size}.params"
        else:
            name = "saved/" + filename + ".params"

        save_params(self.parameters(), name)

    def parameters(self):
        return self.W_f, self.b_f, self.W_i, self.b_i, self.W_g, self.b_g, self.W_o, self.b_o, self.W_v, self.b_v

    def forward(self, inputs, h_prev, C_prev):

        W_f, b_f, W_i, b_i, W_g, b_g, W_o, b_o, W_v, b_v = self.parameters()

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
        W_f, b_f, W_i, b_i, W_g, b_g, W_o, b_o, W_v, b_v = self.parameters()

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

        grads = W_f_d, b_f_d, W_i_d, b_i_d, W_g_d, b_g_d, W_o_d, b_o_d, W_v_d, b_v_d
        grads = clip_gradient_norm(grads)

        return loss, grads

    def update(self, grads, lr=1e-3):  # Take a step
        params = self.parameters()
        for param, grad in zip(params, grads):
            param -= lr * grad

    def train(self, data, num_epochs, lr):
        data_len = len(data)
        name = f"saved/LSTM_Parameters_H{self.hidden_size}_V{self.vocab_size}.params"
        
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
            print_epoch(i + 1, current_epoch_loss, spam_detected,
                        spam_undetected, ham_detected, ham_undetected, data_len)

            # Auto saves every 20 epochs during a training session
            if ((i + 1) % 20 == 0):
                save_params(self.parameters(), name)
                


