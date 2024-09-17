''' 
first one is TPR or True Positive Rate, which is the same as recall.
TPR = TP / (TP + FN)
'''
from matplotlib import pyplot as plt
from appendix.evaluation.classification.accuracy_v2 import false_positive, recall, true_negative, true_positive


def tpr(y_true, y_pred):
    """
    Function to calculate tpr
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: tpr/recall
    """
    return recall(y_true, y_pred)

# TPR or recall is also known as sensitivity.
# And FPR or False Positive Rate, which is defined as:
# FPR = FP / (TN + FP)
def fpr(y_true, y_pred):
    """
    Function to calculate fpr
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: fpr
    """
    fp = false_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    return fp / (tn + fp)

'''
And 1 - FPR is known as specificity or True Negative Rate or TNR.
Let’s assume that we have only 15 samples and their target values are binary:
Actual targets : [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1]
We train a model like the random forest, and we can get the probability of when a
sample is positive.
Predicted probabilities for 1: [0.1, 0.3, 0.2, 0.6, 0.8, 0.05, 0.9, 0.5, 0.3, 0.66, 0.3,
0.2, 0.85, 0.15, 0.99]
For a typical threshold of >= 0.5, we can evaluate all the above values of precision,
recall/TPR, F1 and FPR. But we can do the same if we choose the value of the
threshold to be 0.4 or 0.6. In fact, we can choose any value between 0 and 1 and
calculate all the metrics described above.
Let’s calculate only two values, though: TPR and FPR.
'''
# empty lists to store tpr
# and fpr values
tpr_list = []
fpr_list = []
# actual targets
y_true = [0, 0, 0, 0, 1, 0, 1,
0, 0, 1, 0, 1, 0, 0, 1]

# predicted probabilities of a sample being 1
y_pred = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05,
0.9, 0.5, 0.3, 0.66, 0.3, 0.2,
0.85, 0.15, 0.99]
# handmade thresholds
thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5,
0.6, 0.7, 0.8, 0.85, 0.9, 0.99, 1.0]

# loop over all thresholds
for thresh in thresholds:
    # calculate predictions for a given threshold
    temp_pred = [1 if x >= thresh else 0 for x in y_pred]
#calculate tpr
temp_tpr = tpr(y_true, temp_pred)
# calculate fpr
temp_fpr = fpr(y_true, temp_pred)
# append tpr and fpr to lists
tpr_list.append(temp_tpr)
fpr_list.append(temp_fpr)

plt.figure(figsize=(7, 7))
plt.fill_between(fpr_list, tpr_list, alpha=0.4)
plt.plot(fpr_list, tpr_list, lw=3)
plt.xlim(0, 1.0)
plt.ylim(0, 1.0)
plt.xlabel('FPR', fontsize=15)
plt.ylabel('TPR', fontsize=15)
plt.show()
'''Figure 4: Receiver operating characteristic (ROC) curve
This curve is also known as the Receiver Operating Characteristic (ROC). And
if we calculate the area under this ROC curve, we are calculating another metric
which is used very often when you have a dataset which has skewed binary targets.
'''

'''This metric is known as the Area Under ROC Curve or Area Under Curve or
just simply AUC. '''

from sklearn import metrics
metrics.roc_auc_score(y_true, y_pred)

'''
AUC values range from 0 to 1.
- AUC = 1 implies you have a perfect model. Most of the time, it means that
you made some mistake with validation and should revisit data processing
and validation pipeline of yours. If you didn’t make any mistakes, then
congratulations, you have the best model one can have for the dataset you
built it on.
- AUC = 0 implies that your model is very bad (or very good!). Try inverting
the probabilities for the predictions, for example, if your probability for the
positive class is p, try substituting it with 1-p. This kind of AUC may also
mean that there is some problem with your validation or data processing.
- AUC = 0.5 implies that your predictions are random. So, for any binary
classification problem, if I predict all targets as 0.5, I will get an AUC of
0.5.
'''

'''
calculating probabilities and AUC, you would want to make predictions on
the test set. Depending on the problem and use-case, you might want to either have
probabilities or actual classes. If you want to have probabilities, it’s effortless. You
already have them. If you want to have classes, you need to select a threshold. In
the case of binary classification, you can do something like the following.
Prediction = Probability >= Threshold
Which means, that prediction is a new list which contains only binary variables. An
item in prediction is 1 if the probability is greater than or equal to a given threshold
else the value is 0.
And guess what, you can use the ROC curve to choose this threshold! The ROC
curve will tell you how the threshold impacts false positive rate and true positive
rate and thus, in turn, false positives and true positives. You should choose the
threshold that is best suited for your problem and datasets.
For example, if you don’t want to have too many false positives, you should have a
high threshold value. This will, however, also give you a lot more false negatives.
Observe the trade-off and select the best threshold. Let’s see how these thresholds
impact true positive and false positive values
'''

# empty lists to store true positive
# and false positive values
tp_list = []
fp_list = []
# actual targets
y_true = [0, 0, 0, 0, 1, 0, 1,
0, 0, 1, 0, 1, 0, 0, 1]
# predicted probabilities of a sample being 1
y_pred = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05,
0.9, 0.5, 0.3, 0.66, 0.3, 0.2,
0.85, 0.15, 0.99]
# some handmade thresholds
thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5,
0.6, 0.7, 0.8, 0.85, 0.9, 0.99, 1.0]
# loop over all thresholds
for thresh in thresholds:
    # calculate predictions for a given threshold
    temp_pred = [1 if x >= thresh else 0 for x in y_pred]
# calculate tp
temp_tp = true_positive(y_true, temp_pred)
# calculate fp
temp_fp = false_positive(y_true, temp_pred)
# append tp and fp to lists
tp_list.append(temp_tp)
fp_list.append(temp_fp)