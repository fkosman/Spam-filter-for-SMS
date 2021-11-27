import numpy as np
import os
import os.path
import sys
import random
from datetime import datetime
random.seed(datetime.now())
from models.header import *

class RNN:
    def __init__(self, h_size, v_size):
        self.hidden_size = h_size
        self.vocab_size = v_size

        # Weight matrix (input to hidden state)
        self.U = np.random.randn(h_size, v_size)

        # Weight matrix (recurrent computation)
        self.V = np.random.randn(h_size, h_size)

        # Weight matrix (hidden state to output)
        self.W = np.random.randn(1, h_size)

        # Bias (hidden state)
        self.b_hidden = np.random.randn(h_size, 1)

        # Bias (output)
        self.b_out = np.random.randn(1, 1)

    def parameters(self):
        return self.U, self.V, self.W, self.b_hidden, self.b_out

    def load(self, data, val_data, start_epoch, num_epochs, lr):
        name = f"saved/RNN_Parameters_H{self.hidden_size}_Epoch{start_epoch - 1}.params"

        if not os.path.isfile(name):
            sys.exit("\nNo parameter file found for given hidden-size & epoch.\n")

        with open(name) as paramfile:
            paramfile.readline()
            for param in self.parameters(): load_matrix(param, paramfile)

        self.train(data, val_data, start_epoch, num_epochs, lr)

    def forward(self, inputs):

        hidden_state = np.zeros((self.hidden_size, 1))

        U, V, W, b_h, b = self.parameters()
        outputs, hidden_states = [], []

        # For each element in input sequence
        for t in range(len(inputs)):
            hidden_state = (U @ inputs[t]) + (V @ hidden_state) + b_h

            # Normalize hidden state
            hidden_state = tanh(hidden_state)

            out = sigmoid((W @ hidden_state) + b)

            outputs.append(out)
            hidden_states.append(hidden_state.copy())

        return outputs, hidden_states

    def backward(self, inputs, outputs, hidden_states, target):
        U, V, W, b_h, b = self.parameters()

        # Initialize gradients as zero
        d_U, d_V, d_W = np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)
        d_b_hidden, d_b_out = np.zeros_like(b_h), np.zeros_like(b)

        # Keep track of hidden state derivative and loss
        d_h_next = np.zeros_like(hidden_states[0])
        loss = 0

        for t in reversed(range(len(outputs))):
            # Add loss
            loss += np.absolute(target - outputs[t])

            # Backpropogate into output
            d_o = -(target - outputs[t]) / np.absolute(target - outputs[t] + 1e-5)

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

    def train(self, data, val_data, start_epoch, num_epochs, lr):
        val_len = len(val_data)

        for i in range(num_epochs):
            current_epoch = start_epoch + i
            random.shuffle(data)
            training_loss = 0

            for x, y in data:
                outputs, hidden_states = self.forward(x)

                loss, grads = self.backward(x, outputs, hidden_states, y)
                training_loss += loss
                self.update(grads, lr)

            validation_loss, spam_detected, spam_undetected, ham_detected, ham_undetected = self.eval(val_data)
            training_loss /= len(data)

            log = epoch_log(i + start_epoch, training_loss[0,0], validation_loss[0,0], spam_detected,
                            spam_undetected, ham_detected, ham_undetected, val_len, lr)
            print(log, end="")
            
            with open(f"logs/RNN_Parameters_H{self.hidden_size}_Epoch{current_epoch - 1}.log", "a") as logfile:
                logfile.write(log)
            os.rename(f"logs/RNN_Parameters_H{self.hidden_size}_Epoch{current_epoch - 1}.log",
                      f"logs/RNN_Parameters_H{self.hidden_size}_Epoch{current_epoch}.log")

            # Auto saves every 20 epochs during a training session
            if ((current_epoch) % 20 == 0):
                save_params(self.parameters(),
                            f"saved/RNN_Parameters_H{self.hidden_size}_Epoch{current_epoch}.params")
                if current_epoch > 20:
                    if current_epoch - (start_epoch - 1) >= 20:
                        os.remove(
                        f"saved/RNN_Parameters_H{self.hidden_size}_Epoch{current_epoch - 20}.params")
                    else:
                        os.remove(
                        f"saved/RNN_Parameters_H{self.hidden_size}_Epoch{start_epoch - 1}.params")

        if current_epoch % 20 != 0:
            save_params(self.parameters(),
                        f"saved/RNN_Parameters_H{self.hidden_size}_Epoch{current_epoch}.params")
            if current_epoch > 20:
                os.remove(
                f"saved/RNN_Parameters_H{self.hidden_size}_Epoch{current_epoch - (current_epoch % 20)}.params")

    def eval(self, data):
        validation_loss = 0
        spam_detected = 0
        spam_undetected = 0
        ham_detected = 0
        ham_undetected = 0

        for x, y in data:
            outputs, _ = self.forward(x)
            result = outputs[len(outputs) - 1]
            for output in outputs:
                validation_loss += np.absolute(y - output)

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

        validation_loss /= len(data)

        return validation_loss, spam_detected, spam_undetected, ham_detected, ham_undetected

