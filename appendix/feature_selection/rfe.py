'''Another greedy approach is known as recursive feature elimination (RFE). In the
previous method, we started with one feature and kept adding new features, but in
RFE, we start with all features and keep removing one feature in every iteration that
provides the least value to a given model. But how to do we know which feature
offers the least value? Well, if we use models like linear support vector machine
(SVM) or logistic regression, we get a coefficient for each feature which decides
the importance of the features. In case of any tree-based models, we get feature
importance in place of coefficients. In each iteration, we can eliminate the least
important feature and keep eliminating it until we reach the number of features
needed. When we are doing recursive feature elimination, in each iteration, we remove the
feature which has the feature importance or the feature which has a coefficient
close to 0. Please remember that when you use a model like logistic regression for
binary classification, the coefficients for features are more positive if they are
important for the positive class and more negative if they are important for the
negative class.'''

import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
# fetch a regression dataset
data = fetch_california_housing()
X = data["data"]
col_names = data["feature_names"]
y = data["target"]

# initialize the model
model = LinearRegression()
# initialize RFE
rfe = RFE(
estimator=model,
n_features_to_select=3
)
# fit RFE
rfe.fit(X, y)
# get the transformed data with
# selected columns
X_transformed = rfe.transform(X)