import numpy as np

def sigmoid(x, derivative=False):
    """
    Computes the element-wise sigmoid activation function for an array x.
    Args:
     `x`: the array where the function is applied
     `derivative`: if set to True will return the derivative instead of the forward pass
    """
    x_safe = x + 1e-15
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
    x_safe = x + 1e-15
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
    clip_coef = max_norm / (total_norm + 1e-15)

    # If the total norm is larger than the maximum allowable norm,
    # then clip the gradient
    if clip_coef < 1:
        for grad in grads:
            grad *= clip_coef

    return grads

def epoch_log(epoch, training_loss, validation_loss, spam_detected,
                spam_undetected, ham_detected, ham_undetected, val_acc, train_acc, lr):

    if epoch == 1: result = ""
    else: result = "\n**********************************************************\n"
    
    result += f"Epoch {epoch}\nLR = {lr}\n"
    result += f"Training loss: {training_loss}\n"
    result += f"Validation loss: {validation_loss}\n"
    result += f"Training accuracy: {train_acc}%\n"
    result += f"Validation accuracy: {val_acc}%\n"
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

def save_params(model):
    file = open("saved/" + model.name + ".params", "w")

    file.write("{:^}".format("**************** Model Parameters ****************"))
    file.write(f"\nEpochs: {model.epochs_trained}")
    file.write(f"\tHidden: {model.hidden_size}")
    file.write(f"\tVocab: {model.vocab_size}")
    for param in model.parameters():
        file.write(matrix_to_string(param))
    file.close()