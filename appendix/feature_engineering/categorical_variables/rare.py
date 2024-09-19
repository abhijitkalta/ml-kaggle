'''
If you have a fixed test set, you can add your test data to training to know about the
categories in a given feature. This is very similar to semi-supervised learning in
which you use data which is not available for training to improve your model. This
will also take care of rare values that appear very less number of times in training
data but are in abundance in test data. Your model will be more robust

A simple concatenation of training and test sets to learn about the categories present in the
test set but not in the training set or rare categories in the training set.

This trick works when you have a problem where you already have the test dataset.
It will not work in live setting
'''

import pandas as pd
from sklearn import preprocessing
# read training data
train = pd.read_csv("../input/cat_train.csv")
#read test data
test = pd.read_csv("../input/cat_test.csv")
# create a fake target column for test data
# since this column doesn't exist
test.loc[:, "target"] = -1
# concatenate both training and test data
data = pd.concat([train, test]).reset_index(drop=True)
# make a list of features we are interested in
# id and target is something we should not encode
features = [x for x in train.columns if x not in ["id", "target"]]
# loop over the features list
for feat in features:
    # create a new instance of LabelEncoder for each feature
    lbl_enc = preprocessing.LabelEncoder()
    # note the trick here
    # since its categorical data, we fillna with a string
    # and we convert all the data to string type
    # so, no matter its int or float, its converted to string
    # int/float but categorical!!!
    temp_col = data[feat].fillna("NONE").astype(str).values
    # we can use fit_transform here as we do not
    # have any extra test data that we need to
    # transform on separately
    data.loc[:, feat] = lbl_enc.fit_transform(temp_col)
# split the training and test data again
train = data[data.target != -1].reset_index(drop=True)
test = data[data.target == -1].reset_index(drop=True)

'''
Another technique -
So, you can either assume that your test data will have the same categories as
training or you can introduce a rare or unknown category to training to take care of
new categories in test data.

We can now define our criteria for calling a value “rare”. Let’s say the requirement
for a value being rare in this column is a count of less than 2000. So, it seems, J and
L can be marked as rare values.
'''
 
df.ord_4 = df.ord_4.fillna("NONE") # type: ignore
df.loc[ # type: ignore
df["ord_4"].value_counts()[df["ord_4"]].values < 2000, # type: ignore
 "ord_4"] = "RARE"