'''In this example, you take an input image and predicts the probability for it being benign
or malignant.
In these kinds of datasets, you might have multiple images for the same patient in
the training dataset. So, to build a good cross-validation system here, you must have
stratified k-folds, but you must also make sure that patients in training data do not
appear in validation data. Fortunately, scikit-learn offers a type of cross-validation
known as GroupKFold.
'''
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.datasets import make_classification

# Load the dataset
data = pd.read_csv('/Users/abhijit/Developer/ml_kaggle/appendix/cross_validation/train.csv')

# Assuming 'target' is the column to stratify and 'group' is the column for grouping
X = data.drop(columns=['target'])
y = data['target']
groups = data['group']

# Initialize StratifiedKFold and GroupKFold
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
group_kfold = GroupKFold(n_splits=5)

# Example of using StratifiedKFold
for train_index, test_index in stratified_kfold.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    # Your training code here

# Example of using GroupKFold
for train_index, test_index in group_kfold.split(X, y, groups):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    # Your training code here
    # Generate a synthetic dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

    # Assuming 'groups' is generated or provided separately
    groups = pd.Series([i // 50 for i in range(1000)])

    # Initialize StratifiedKFold and GroupKFold
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    group_kfold = GroupKFold(n_splits=5)

    # Example of using StratifiedKFold
    for train_index, test_index in stratified_kfold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Your training code here

    # Example of using GroupKFold
    for train_index, test_index in group_kfold.split(X, y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Your training code here