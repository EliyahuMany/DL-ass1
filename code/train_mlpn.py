import mlpn as ml
import utils as ut
import random
import numpy as np

STUDENT = {'name': 'Eliyahu Many_Yarin Ifrach',
           'ID': '308249150_205697410'}


def feats_to_vec(features):
    # YOUR CODE HERE.
    # # Should return a numpy vector of features.
    feats_vec = np.zeros(len(ut.F2I))
    for bigram in features:
        if bigram in ut.F2I:
            feats_vec[ut.F2I[bigram]] += 1
    # normalization
    num_of_matches = np.sum(feats_vec)
    return np.divide(feats_vec, num_of_matches)


def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        x = feats_to_vec(features)  # convert features to a vector.
        y = ut.L2I[label]  # convert the label to number if needed.
        pred = ml.predict(x, params)
        if (y == pred):
            good += 1
        else:
            bad += 1
            # Compute the accuracy (a scalar) of the current parameters
            # on the dataset.
            # accuracy is (correct_predictions / all_predictions)
    return good / (good + bad)


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.
    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in xrange(num_iterations):
        cum_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features)  # convert features to a vector.
            y = ut.L2I[label]  # convert the label to number if needed.
            loss, grads = ml.loss_and_gradients(x, y, params)
            cum_loss += loss
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.
            for i, param in enumerate(params):
                param -= learning_rate* grads[i]

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print I, train_loss, train_accuracy, dev_accuracy
    return params


def test(test_data, params):
    prediction_file= open("data/test.pred", 'w')
    for label, features in test_data:
        x = feats_to_vec(features)  # convert features to a vector.
        pred = ml.predict(x, params)
        for key, val in ut.L2I.items():
            if val == pred:
                label = key
                break
        prediction_file.write(str(label) + "\n")
    prediction_file.close()


if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.

    num_iterations = 50
    learning_rate = 0.01
    hidden1 = 20
    hidden2 = 30
    dims = [len(ut.F2I), hidden1, hidden2, len(ut.L2I)]

    params = ml.create_classifier(dims)
    trained_params = train_classifier(ut.TRAIN, ut.DEV, num_iterations, learning_rate, params)
    test(ut.TRAIN,trained_params)
