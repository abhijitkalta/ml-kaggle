'''
Error = True Value – Predicted Value
Absolute error is just absolute of the above.
Absolute Error = Abs ( True Value – Predicted Value )
Then we have mean absolute error (MAE). It’s just mean of all absolute errors.
'''

import numpy as np
def mean_absolute_error(y_true, y_pred):
    """
    This function calculates mae
    :param y_true: list of real numbers, true values
    :param y_pred: list of real numbers, predicted values
    :return: mean absolute error
    """
    # initialize error at 0
    error = 0
    # loop over all samples in the true and predicted list
    for yt, yp in zip(y_true, y_pred):
    # calculate absolute error
    # and add to error
        error += np.abs(yt - yp)
    # return mean error
    return error / len(y_true)

'''
Similarly, we have squared error and mean squared error (MSE).
Squared Error = ( True Value – Predicted Value )2
'''

def mean_squared_error(y_true, y_pred):
    """
    This function calculates mse
    :param y_true: list of real numbers, true values
    :param y_pred: list of real numbers, predicted values
    :return: mean squared error
    """
    # initialize error at 0
    error = 0
    # loop over all samples in the true and predicted list
    for yt, yp in zip(y_true, y_pred):
    # calculate squared error
    # and add to error
        error += (yt - yp) ** 2
    # return mean error
    return error / len(y_true)

'''
MSE and RMSE (root mean squared error) are the most popular metrics used in
evaluating regression models.
RMSE = SQRT ( MSE )
'''

'''
Another type of error in same class is squared logarithmic error. Some people
call it SLE, and when we take mean of this error across all samples, it is known as
MSLE (mean squared logarithmic error) and implemented as follows.
'''

import numpy as np
def mean_squared_log_error(y_true, y_pred):
    """
    This function calculates msle
    :param y_true: list of real numbers, true values
    :param y_pred: list of real numbers, predicted values
    :return: mean squared logarithmic error
    Approaching (Almost) Any Machine Learning Problem
    68
    """
    # initialize error at 0
    error = 0
    # loop over all samples in true and predicted list
    for yt, yp in zip(y_true, y_pred):
    # calculate squared log error
    # and add to error
        error += (np.log(1 + yt) - np.log(1 + yp)) ** 2
    # return mean error
    return error / len(y_true)


'''Percentage Error = ( ( True Value – Predicted Value ) / True Value ) * 100
Same can be converted to mean percentage error for all samples.
'''
def mean_percentage_error(y_true, y_pred):
    """
    This function calculates mpe
    :param y_true: list of real numbers, true values
    :param y_pred: list of real numbers, predicted values
    :return: mean percentage error
    """
    # initialize error at 0
    error = 0
    # loop over all samples in true and predicted list
    for yt, yp in zip(y_true, y_pred):
    # calculate percentage error
    # and add to error
        error += (yt - yp) / yt
    # return mean percentage error
    return error / len(y_true)