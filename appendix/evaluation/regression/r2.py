'''Let’s talk about another regression metric known as R2 (R-squared), also known
as the coefficient of determination.
In simple words, R-squared says how good your model fits the data. R-squared
closer to 1.0 says that the model fits the data quite well, whereas closer 0 means
that model isn’t that good. R-squared can also be negative when the model just
makes absurd predictions.
'''
import numpy as np
def r2(y_true, y_pred):
    """
    This function calculates r-squared score
    :param y_true: list of real numbers, true values
    :param y_pred: list of real numbers, predicted values
    :return: r2 score
    """
    # calculate the mean value of true values
    mean_true_value = np.mean(y_true)
    # initialize numerator with 0
    numerator = 0
    # initialize denominator with 0
    denominator = 0
    # loop over all true and predicted values
    for yt, yp in zip(y_true, y_pred):
    # update numerator
        numerator += (yt - yp) ** 2
    # update denominator
        denominator += (yt - mean_true_value) ** 2
    # calculate the ratio
    ratio = numerator / denominator
    # return 1 - ratio
    return 1 - ratio

'''One of them which is quite widely used is quadratic weighted kappa, also known
as QWK. It is also known as Cohen’s kappa. QWK measures the “agreement”
between two “ratings”. The ratings can be any real numbers in 0 to N. And
predictions are also in the same range. An agreement can be defined as how close
these ratings are to each other. So, it’s suitable for a classification problem with N
different categories/classes. If the agreement is high, the score is closer towards 1.0.
In the case of low agreement, the score is close to 0. Cohen’s kappa has a good
implementation in scikit-learn, and detailed discussion of this metric is beyond the
scope of this book.
═════════════════════════════════════════════════════════════════════════
In [X]: from sklearn import metrics
In [X]: y_true = [1, 2, 3, 1, 2, 3, 1, 2, 3]
In [X]: y_pred = [2, 1, 3, 1, 2, 3, 3, 1, 2]
In [X]: metrics.cohen_kappa_score(y_true, y_pred, weights="quadratic")
Out[X]: 0.33333333333333337
In [X]: metrics.accuracy_score(y_true, y_pred)
Out[X]: 0.4444444444444444
═════════════════════════════════════════════════════════════════════════
You can see that even though accuracy is high, QWK is less. A QWK greater than
0.85 is considered to be very good'''