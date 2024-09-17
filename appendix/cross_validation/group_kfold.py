from sklearn.model_selection import GroupKFold, StratifiedKFold
"""
Perform a nested cross-validation using GroupKFold and StratifiedKFold.

In these kinds of datasets, you might have multiple images for the same patient in
the training dataset. So, to build a good cross-validation system here, you must have
stratified k-folds, but you must also make sure that patients in training data do not
appear in validation data. Fortunately, scikit-learn offers a type of cross-validation
known as GroupKFold. Here the patients can be considered as groups. 

This function first splits the data into training and testing sets using GroupKFold,
ensuring that the same group is not represented in both the training and testing sets.
Then, it further splits the training set into training and validation sets using StratifiedKFold,
ensuring that the class distribution is preserved in both sets.

Parameters:
-----------
X : array-like, shape (n_samples, n_features)
    The input samples.
y : array-like, shape (n_samples,)
    The target values.
groups : array-like, shape (n_samples,)
    Group labels for the samples used while splitting the dataset into train/test set.
n_splits : int, default=5
    Number of folds. Must be at least 2.

Yields:
-------
X_train : array-like, shape (n_train_samples, n_features)
    The training input samples.
X_val : array-like, shape (n_val_samples, n_features)
    The validation input samples.
X_group_test : array-like, shape (n_test_samples, n_features)
    The testing input samples.
y_train : array-like, shape (n_train_samples,)
    The training target values.
y_val : array-like, shape (n_val_samples,)
    The validation target values.
y_group_test : array-like, shape (n_test_samples,)
    The testing target values.

Example:
--------
>>> X, y, groups = np.array(...), np.array(...), np.array(...)
>>> for X_train, X_val, X_test, y_train, y_val, y_test in group_stratified_kfold(X, y, groups):
>>>     # Train and evaluate your model
"""
import numpy as np

def group_stratified_kfold(X, y, groups, n_splits=5):
    group_kfold = GroupKFold(n_splits=n_splits)
    stratified_kfold = StratifiedKFold(n_splits=n_splits)

    for group_train_idx, group_test_idx in group_kfold.split(X, y, groups):
        X_group_train, X_group_test = X[group_train_idx], X[group_test_idx]
        y_group_train, y_group_test = y[group_train_idx], y[group_test_idx]

        for strat_train_idx, strat_test_idx in stratified_kfold.split(X_group_train, y_group_train):
            X_train, X_val = X_group_train[strat_train_idx], X_group_train[strat_test_idx]
            y_train, y_val = y_group_train[strat_train_idx], y_group_train[strat_test_idx]

            yield X_train, X_val, X_group_test, y_train, y_val, y_group_test

# Example usage:
# X, y, groups = np.array(...), np.array(...), np.array(...)
# for X_train, X_val, X_test, y_train, y_val, y_test in group_stratified_kfold(X, y, groups):
#     # Train and evaluate your model