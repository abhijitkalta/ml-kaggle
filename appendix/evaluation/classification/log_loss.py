'''
Another important metric you should learn after learning AUC is log loss. In case
of a binary classification problem, we define log loss as:
Log Loss = - 1.0 * ( target * log(prediction) + (1 - target) * log(1 - prediction) )
Where target is either 0 or 1 and prediction is a probability of sample beloonging to class 1
One thing to remember is that log loss penalizes quite
high for an incorrect or a far-off prediction, i.e. log loss punishes you for being very
sure and very wrong.
'''

from sklearn.metrics import logg_loss
logg_loss(y_true, y_pred) # type: ignore

import numpy as np
def log_loss(y_true, y_proba):
    """
    Function to calculate fpr
    :param y_true: list of true values
    :param y_proba: list of probabilities for 1
    :return: overall log loss
    """
    # define an epsilon value
    # this can also be an input
    # this value is used to clip probabilities
    epsilon = 1e-15
    # initialize empty list to store
    # individual losses
    loss = []
    # loop over all true and predicted probability values
    for yt, yp in zip(y_true, y_proba):
        # adjust probability
        # 0 gets converted to 1e-15
        # 1 gets converted to 1-1e-15
        # Why? Think about it!
        yp = np.clip(yp, epsilon, 1 - epsilon)
        # calculate loss for one sample
        temp_loss = - 1.0 * (
        yt * np.log(yp)
        + (1 - yt) * np.log(1 - yp)
        )
    # add to loss list
    loss.append(temp_loss)
    # return mean loss over all samples
    return np.mean(loss)