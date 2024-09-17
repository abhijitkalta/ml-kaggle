import matplotlib.pyplot as plt
from accuracy_v2 import precision, recall
'''
Most of the models predict a probability, and when we predict, we usually choose
this threshold to be 0.5. This threshold is not always ideal, and depending on this
threshold, your value of precision and recall can change drastically. If for every
threshold we choose, we calculate the precision and recall values, we can create a
plot between these sets of values. This plot or curve is known as the precision-recall
curve.
'''

y_true = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
y_pred = [0.02638412, 0.11114267, 0.31620708,
 0.0490937, 0.0191491, 0.17554844,
 0.15952202, 0.03819563, 0.11639273,
 0.079377, 0.08584789, 0.39095342,
 0.27259048, 0.03447096, 0.04644807,
 0.03543574, 0.18521942, 0.05934905,
0.61977213, 0.33056815]

'''
So, y_true is our targets, and y_pred is the probability values for a sample being
assigned a value of 1. So, now, we look at probabilities in prediction instead of the
predicted value (which is most of the time calculated with a threshold at 0.5).
'''
precisions = []
recalls = []
# how we assumed these thresholds is a long story
thresholds = [0.0490937 , 0.05934905, 0.079377,
0.08584789, 0.11114267, 0.11639273,
0.15952202, 0.17554844, 0.18521942,
0.27259048, 0.31620708, 0.33056815,
0.39095342, 0.61977213]
# for every threshold, calculate predictions in binary
# and append calculated precisions and recalls
# to their respective lists
for i in thresholds:
    temp_prediction = [1 if x >= i else 0 for x in y_pred]
    p = precision(y_true, temp_prediction)
    r = recall(y_true, temp_prediction)
    precisions.append(p)
    recalls.append(r)
# Now, we can plot these values of precisions and recalls.
plt.figure(figsize=(7, 7))
plt.plot(recalls, precisions)
plt.xlabel('Recall', fontsize=15)
plt.ylabel('Precision', fontsize=15)

# Choose a threshold that gives both good precision and recall

