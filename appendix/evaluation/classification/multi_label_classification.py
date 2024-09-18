''' Precision at k. If you have a list of original classes for a given
sample and list of predicted classes for the same, precision is defined as the number
of hits in the predicted list considering only top-k predictions, divided by k.
If that’s confusing, it will become apparent with python code.
'''

def pk(y_true, y_pred, k):
    """
    This function calculates precision at k
    for a single sample
    :param y_true: list of values, actual classes
    :param y_pred: list of values, predicted classes
    :return: precision at a given value k
    """
    # if k is 0, return 0. we should never have this
    # as k is always >= 1
    if k == 0:
        return 0
    # we are interested only in top-k predictions
    y_pred = y_pred[:k]
    # convert predictions to set
    pred_set = set(y_pred)
    # convert actual values to set
    true_set = set(y_true)
    # find common values
    common_values = pred_set.intersection(true_set)
    # return length of common values over k
    return len(common_values) / len(y_pred[:k])

'''
Now, we have average precision at k or AP@k. AP@k is calculated using P@k.
For example, if we have to calculate AP@3, we calculate AP@1, AP@2 and AP@3
and then divide the sum by 3.
'''
def apk(y_true, y_pred, k):
    """
    This function calculates average precision at k
    for a single sample
    :param y_true: list of values, actual classes
    :param y_pred: list of values, predicted classes
    :return: average precision at a given value k
    """
    # initialize p@k list of values
    pk_values = []
    # loop over all k. from 1 to k + 1
    for i in range(1, k + 1):
        # calculate p@i and append to list
        pk_values.append(pk(y_true, y_pred, i))
    # if we have no values in the list, return 0
    if len(pk_values) == 0:
        return 0
    # else, we return the sum of list over length of list
    return sum(pk_values) / len(pk_values)

'''
we are interested in all samples, and that’s why we have mean average precision
at k or MAP@k. MAP@k is just an average of AP@k and can be calculated easily
by the following python code.
'''

def mapk(y_true, y_pred, k):
    """
    This function calculates mean avg precision at k
    for a single sample
    :param y_true: list of values, actual classes
    :param y_pred: list of values, predicted classes
    :return: mean avg precision at a given value k
    """
    # initialize empty list for apk values
    apk_values = []
    # loop over all samples
    for i in range(len(y_true)):
    # store apk values for every sample
        apk_values.append(
        apk(y_true[i], y_pred[i], k=k)
        )
    # return mean of apk values list
    return sum(apk_values) / len(apk_values)

#P@K, ap@k, map@k all range from 0 to 1, with 1 being the best

'''
Mean Precision at K (MP@K):


To calculate MP@K, we evaluate the precision of the top K predicted genres.


| Movie   | Relevant Genres | Predicted Genres | Precision |
|---------|-----------------|-----------------|-----------|
| Movie1  | Action, Drama   | Action, Drama, Thriller | 2/3    |
| Movie2  | Comedy, Drama   | Comedy, Drama, Romance  | 2/3    |
| Movie3  | Action, Comedy  | Action, Comedy, Thriller | 2/3    |


MP@K = (2/3 + 2/3 + 2/3) / 3 = 0.67
'''
'''
*Log Loss for Multi-Label Classification*

In multi-label classification, each sample can have multiple labels (e.g., a text can be classified as both "sports" and "news"). The goal is to predict the probability of each label.

*Formula:*

The log loss function for multi-label classification is:

`L(y, y_pred) = - (1/n) * ∑[y[i] * log(y_pred[i]) + (1-y[i]) * log(1-y_pred[i])]`

where:

* `y` is the true label vector (binary vector where 1 indicates presence of label and 0 indicates absence)
* `y_pred` is the predicted probability vector
* `n` is the number of labels
* `i` indexes each label

*Breakdown:*

For each label, the log loss calculates two terms:

1. `y[i] * log(y_pred[i])`: penalty for false negatives (when the model predicts a low probability for a true label)
2. `(1-y[i]) * log(1-y_pred[i])`: penalty for false positives (when the model predicts a high probability for a false label)

The logarithmic function maps the predicted probabilities to a loss value, where:

* A predicted probability close to 1 (true label) results in a low loss
* A predicted probability close to 0 (false label) results in a high loss

*Interpretation:*

The log loss value indicates the difference between predicted probabilities and true labels. A lower log loss value indicates better performance.

*Range:*

Log loss ranges from 0 to infinity, where:

* 0: perfect predictions (all labels correctly classified)
* Infinity: worst possible predictions (all labels incorrectly classified)

*Example Python Implementation:*
```
import numpy as np

def log_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred_clip = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = - np.mean(y_true * np.log(y_pred_clip) + (1 - y_true) * np.log(1 - y_pred_clip))
    return loss

# Example usage
y_true = np.array([[1, 0, 1], [0, 1, 0]])  # true labels
y_pred = np.array([[0.8, 0.2, 0.9], [0.1, 0.9, 0.1]])  # predicted probabilities

log_loss_value = log_loss(y_true, y_pred)
print("Log Loss:", log_loss_value)

np.clip is used to clip the predicted probabilities y_pred to a specific range, preventing extreme values that could lead to numerical instability or division-by-zero errors.

Why clipping is necessary:

1. Logarithmic function: The log loss formula involves the logarithmic function, which approaches negative infinity as its input approaches zero. Clipping ensures that the input to the logarithm is never exactly zero.
2. Division by zero: When calculating log(1 - y_pred), if y_pred is exactly 1, the expression would evaluate to log(0), which is undefined.
```
'''