from header import *

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

    def load(self, filename=None):
        if filename is None:
            name = f"RNN_Parameters_H{self.hidden_size}_V{self.vocab_size}.txt"
        else:
            name = filename

        h_size = self.hidden_size
        v_size = self.vocab_size

        paramfile = open(name)
        for i in range(3): paramfile.readline()
        for param in self.parameters(): load_matrix(param, paramfile)
        paramfile.close()

    def forward(self, inputs):

        hidden_state = np.zeros((self.hidden_size, 1))

        U, V, W, b_h, b = self.parameters()
        outputs, hidden_states = [], []

        # For each element in input sequence
        for t in range(len(inputs)):
            hidden_state = (U @ inputs[t]) + (V @ hidden_state) + b_h

            # Normalize hidden state
            # hidden_state = hidden_state - hidden_state.mean()
            # hidden_state = hidden_state / (hidden_state.std() + 1e-5)
            hidden_state = tanh(hidden_state)

            out = sigmoid((W @ hidden_state) + b)

            outputs.append(out)
            hidden_states.append(hidden_state.copy())
            # print(f"Hidden state before: {hidden_state}")
            # hidden_state = normalize(hidden_state, 2)
            # print(f"Hidden state after: {hidden_state}")
            # hidden_state /= np.linalg.norm(hidden_state)
            # hidden_state = hidden_state * 2

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

                outputs, hidden_states = self.forward(x)
                result = outputs[len(outputs) - 1]

                loss, grads = self.backward(x, outputs, hidden_states, y)

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
            print("{:<32}{:>4}".format(" misclassified regular messages: ", ham_undetected))
            print("{:<38}{:>4}, ".format("Correctly classified spam messages: ", spam_detected), end='')
            print("{:<32}{:>4}".format(" misclassified spam messages: ", spam_undetected))
            print("*****************************************************")
            current_epoch_loss = 0
            spam_detected = 0
            spam_undetected = 0
            ham_detected = 0
            ham_undetected = 0

            if ((i + 1) % 50 == 0):
                U, V, W, b_hidden, b_out = self.parameters()
                file = open(f"RNN_Parameters_H{self.hidden_size}_V{self.vocab_size}.txt", "w")
                file.write(" ************* RNN Model Parameters ************* \n")

                file.write("\nU matrix:\n")
                file.write(matrix_to_string(U))
                file.write("\nV matrix:\n")
                file.write(matrix_to_string(V))
                file.write("\nW matrix:\n")
                file.write(matrix_to_string(W))
                file.write("\nb_hidden matrix:\n")
                file.write(matrix_to_string(b_hidden))
                file.write("\nb_out matrix:\n")
                file.write(matrix_to_string(b_out))
                file.close()