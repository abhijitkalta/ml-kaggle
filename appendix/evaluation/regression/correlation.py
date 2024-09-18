from appendix.evaluation.classification.accuracy_v2 import false_negative, false_positive, true_negative, true_positive
'''An important metric is Matthew’s Correlation Coefficient (MCC). MCC ranges
from -1 to 1. 1 is perfect prediction, -1 is imperfect prediction and 0 is random'''

def mcc(y_true, y_pred):
    """
    This function calculates Matthew's Correlation Coefficient
    for binary classification.
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: mcc score
    """
    tp = true_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    numerator = (tp * tn) - (fp * fn)
    denominator = (
    (tp + fp) *
    (fn + tn) *
    (fp + tn) *
    (tp + fn)
    )
    denominator = denominator ** 0.5
    return numerator/denominator