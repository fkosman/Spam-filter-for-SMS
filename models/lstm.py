from vocabulary import *
from header import *
import numpy as np
import os
import os.path
import sys
import random
from datetime import datetime
random.seed(datetime.now())

class LSTM:

    def __init__(self, h_size, v_size, name, load=False):
        if load:
            self.name = name
            if not os.path.isfile("../saved/" + self.name + ".params"):
                sys.exit("\nNo parameter file found for this model.\n")

            with open("../saved/" + self.name + ".params") as paramfile:
                paramfile.readline()
                vals = paramfile.readline().strip().split()
                self.epochs_trained = int(vals[1])
                h_size = int(vals[3])
                v_size = int(vals[5])
            self.hidden_size = h_size
            self.vocab_size = v_size
            z_size = h_size + v_size
        else:
            self.hidden_size = h_size
            self.vocab_size = v_size
            if name:
                self.name = name
            else:
                self.name = f"LSTM_{self.hidden_size}Hidden_{self.vocab_size}Vocab"
            self.epochs_trained = 0
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

        if load:
            self.load()

    def save(self):
        if os.path.isfile("../saved/" + self.name + ".params"):
            print("\nA parameter file for a model with the same architecture already exists.")
            ans = input("Do you want to overwrite it? (y/n): ")

            while ans.lower() != "y" and ans.lower() != "n" and ans.lower() != "yes" and ans.lower() != "no":
                print("Invalid input.")
                print("A parameter file with the name \"" + self.name + "\" already exists.")
                ans = input("Do you want to overwrite it? (y/n): ")
            if ans.lower() == "y" or ans.lower() == "yes":
                save_params(self)
            else:
                sys.exit("\nSave failed.\n")

        save_params(self)

    def load(self):
        with open("../saved/" + self.name + ".params") as paramfile:
            paramfile.readline()
            paramfile.readline()
            for param in self.parameters(): load_matrix(param, paramfile)

    def parameters(self):
        return self.W_f, self.b_f, self.W_i, self.b_i, self.W_g, self.b_g, self.W_o, self.b_o, self.W_v, self.b_v

    def forward(self, inputs):
        
        W_f, b_f, W_i, b_i, W_g, b_g, W_o, b_o, W_v, b_v = self.parameters()

        # Save a list of computations for each of the components in the LSTM
        x_s, z_s, f_s, i_s, = [], [], [], []
        g_s, C_s, o_s, h_s = [], [], [], []
        v_s, output_s = [], []

        # Append the initial cell and hidden state to their respective lists
        h_prev = np.zeros((self.hidden_size, 1))
        C_prev = np.zeros((self.hidden_size, 1))
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

            # Calculate activation
            output = tanh(v)
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

        result = outputs[len(outputs) - 1]
        loss = 10 * np.absolute(target - result)

        for t in reversed(range(len(outputs))):
            dv = -(target - outputs[t]) / (np.absolute(target - outputs[t]) + 1e-15)
            dv = dv * tanh(v[t], derivative=True)

            # Update the gradient of the relation of the hidden-state to the output gate
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

            # Get the previous hidden cell state
            C_prev = C[t - 1]
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

    def train(self, data, val_data, num_epochs, lr):
        val_len = len(val_data)
        train_len = len(data)

        for i in range(num_epochs):
            self.epochs_trained += 1
            random.shuffle(data)
            training_loss = 0
            num_correct = 0

            for x, y in data:
                z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs = self.forward(x)
                result = outputs[len(outputs) - 1]
                if y == 1 and result >= 0:
                    num_correct += 1
                if y == -1 and result < 0:
                    num_correct += 1

                loss, grads = self.backward(z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs, y)
                training_loss += loss
                self.update(grads, lr)

            validation_loss, spam_detected, spam_undetected, ham_detected, ham_undetected = self.eval(val_data)
            validation_accuracy = 100 * (ham_detected + spam_detected) / val_len

            training_loss /= train_len
            training_accuracy = 100 * num_correct / train_len

            log = epoch_log(self.epochs_trained, training_loss[0,0], validation_loss[0,0], spam_detected,
                        spam_undetected, ham_detected, ham_undetected, validation_accuracy, training_accuracy, lr)
            print(log, end="")

            with open(f"../logs/{self.name}.log", "a") as logfile:
                logfile.write(log)

            # Auto saves every 20 epochs during a training session
            if ((self.epochs_trained) % 20 == 0):
                save_params(self)

        save_params(self)

    def eval(self, data):
        validation_loss = 0
        spam_detected = 0
        spam_undetected = 0
        ham_detected = 0
        ham_undetected = 0

        for x, y in data:
            h = np.zeros((self.hidden_size, 1))
            c = np.zeros((self.hidden_size, 1))
            _, _, _, _, _, _, _, _, outputs = self.forward(x)

            result = outputs[len(outputs) - 1]
            if y == 1:
                if result >= 0:
                    ham_detected += 1
                else:
                    ham_undetected += 1
            else:
                if result < 0:
                    spam_detected += 1
                else:
                    spam_undetected += 1

            validation_loss += 10 * np.absolute(y - result)

        validation_loss /= len(data)

        return validation_loss, spam_detected, spam_undetected, ham_detected, ham_undetected

