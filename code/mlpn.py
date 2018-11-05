import numpy as np
import loglinear as ll

STUDENT = {'name': 'Eliyahu Many_Yarin Ifrach',
           'ID': '308249150_205697410'}


def classifier_output(x, params):
    h = x
    for i in range(0, len(params), 2):
        z = np.dot(h, params[i]) + params[i + 1]
        h = np.tanh(z)
    return ll.softmax(z)


def predict(x, params):
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    (of course, if we request a linear classifier (ie, params is of length 2),
    you should not have gW2 and gb2.)
    """
    temp = x
    probs = classifier_output(x, params)
    y1 = np.zeros(len(probs))
    y1[y] = 1
    h = []
    z = []
    grad = []
    gradients = []
    loss = -1 * np.log(probs[y])
    grad_temp = -1 * (y1 - probs)

    h.append(temp)
    for i in range(0, len(params), 2):
        z2 = np.dot(temp, params[i]) + params[i + 1]
        z.append(z2)
        temp = np.tanh(z2)
        h.append(temp)
    h.pop()
    z.pop()

    gradients.append(np.outer(h.pop(), grad_temp))
    gradients.append(np.copy(grad_temp))

    for i, (w, b) in enumerate(zip(params[-2::-2], params[-1::-2])):
        if (len(z) != 0):
            z1 = z.pop()
            w_temp = w
            if (len(h) != 0):
                h_temp = h.pop()

                dz_dh = w_temp
                dh_dz = 1 - np.square(np.tanh(z1))
                dz_dw = h_temp

                grad_temp = np.dot(grad_temp, np.transpose(dz_dh)) * dh_dz
                gradients.append(np.outer(dz_dw, grad_temp))
                gradients.append(np.copy(grad_temp))
    for w, b in zip(gradients[0::2], gradients[1::2]):
        grad.append(b)
        grad.append(w)

    return loss, list(reversed(grad))


def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.

    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    print dims
    params = []
    for dim1, dim2 in zip(dims, dims[1:]):
        eq1 = np.sqrt(6) / (np.sqrt(dim1 + dim2))
        eq2 = np.sqrt(6) / (np.sqrt(dim2))

        params.append(np.random.uniform(-eq1, eq1, [dim1, dim2]))
        params.append(np.random.uniform(-eq2, eq2, dim2))
    return params
