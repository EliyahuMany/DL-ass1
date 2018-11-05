import numpy as np
import loglinear as ll

STUDENT = {'name': 'Eliyahu Many_Yarin Ifrach',
           'ID': '308249150_205697410'}


def classifier_output(x, params):
    W = params[0]
    b = params[1]
    U = params[2]
    b_tag = params[3]
    eq = np.dot(U, (np.tanh(np.dot(W, x) + b))) + b_tag

    return ll.softmax(eq)


def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """
    probs = classifier_output(x, params)
    y1 = np.zeros(len(probs))
    y1[y] = 1
    gb_tag = -1 * (y1 - probs)

    W = params[0]
    b = params[1]
    U = params[2]
    b_tag = params[3]

    gb = np.dot(gb_tag, U) * (1 - np.square(np.tanh(np.dot(W, x) + b)))
    gW = np.outer(gb, x)
    gU = np.outer(gb_tag, np.tanh(np.dot(W, x) + b))

    loss = -1 * np.log(probs[y])

    return loss, [gW, gb, gU, gb_tag]


def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    eq1 = np.sqrt(6) / (np.sqrt(hid_dim + in_dim))
    eq2 = np.sqrt(6) / (np.sqrt(hid_dim))

    eq3 = np.sqrt(6) / (np.sqrt(out_dim + hid_dim))
    eq4 = np.sqrt(6) / (np.sqrt(out_dim))

    W = np.random.uniform(-eq1, eq1, [hid_dim, in_dim])
    b = np.random.uniform(-eq2, eq2, hid_dim)
    U = np.random.uniform(-eq3, eq3, [out_dim, hid_dim])
    b_tag = np.random.uniform(-eq4, eq4, out_dim)

    params = [W, b, U, b_tag]
    return params
