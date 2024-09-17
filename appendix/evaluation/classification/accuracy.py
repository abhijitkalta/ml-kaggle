'''When we have an equal number of positive and negative samples in a binary
classification metric, we generally use accuracy, precision, recall and f1.
Accuracy: It is one of the most straightforward metrics used in machine learning.
It defines how accurate your model is. For the problem described above, if you build
a model that classifies 90 images accurately, your accuracy is 90% or 0.90. If only
83 images are classified correctly, the accuracy of your model is 83% or 0.83.'''

def accuracy(y_true, y_pred):
    """
    Function to calculate accuracy
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: accuracy score
    """
    # initialize a simple counter for correct predictions
    correct_counter = 0
    # loop over all elements of y_true
    # and y_pred "together"
    for yt, yp in zip(y_true, y_pred):
        # if the prediction is correct, increase the counter
        if yt == yp:
        # if prediction is equal to truth, increase the counter
            correct_counter += 1
    # return accuracy
    # which is correct predictions over the number of samples
    return correct_counter / len(y_true)