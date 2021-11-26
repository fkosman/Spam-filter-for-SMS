import numpy as np

vocab_size = 30

def char_to_index(c):
    # Number of characters in our vocabulary is 30
    # (26 letters + punctuation + numbers + white space + all others)
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
        if encoded[i, 0] == 1:
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

def epoch_log(epoch, training_loss, validation_loss, spam_detected,
                spam_undetected, ham_detected, ham_undetected, data_len, lr):

    if epoch == 1: result = ""
    else: result = "\n**********************************************************\n"
    
    result += f"Epoch {epoch}\nLR = {lr}\n"
    result += f"Training loss: {training_loss}\n"
    result += f"Validation loss: {validation_loss}\n"
    result += f"Prediction accuracy: {100 * (ham_detected + spam_detected) / data_len}%\n"
    result += "{:<40}{:>4}\n".format("Correctly classified regular messages: ", ham_detected)
    result += "{:<40}{:>4}\n".format("Misclassified regular messages: ", ham_undetected)
    result += "{:<40}{:>4}\n".format("Correctly classified spam messages: ", spam_detected)
    result += "{:<40}{:>4}".format("Misclassified spam messages: ", spam_undetected)

    return result

def matrix_to_string(mat):
    result = ""
    rows, cols = mat.shape

    for row in range(rows):
        result += "\n"
        for col in range(cols):
            result += str(mat[row, col])
            result += "\t"

    return result

def load_matrix(mat, file):
    rows, cols = mat.shape

    for row in range(rows):
        values = file.readline().strip().split("\t")
        for col in range(cols):
            mat[row, col] = float(values[col])

def save_params(params, name):
    file = open(name, "w")

    file.write("{:^}".format("**************** Model Parameters ****************"))
    for param in params:
        file.write(matrix_to_string(param))
    file.close()