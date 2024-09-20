'''You can choose
features from one model and use another model to train. For example, you can use
Logistic Regression coefficients to select the features and then use Random Forest
to train the model on chosen features. Scikit-learn also offers SelectFromModel
class that helps you choose features directly from a given model. You can also
specify the threshold for coefficients or feature importance if you want and the
maximum number of features you want to select.
Approaching (Almost) Any Machine Learning Problem
165
Take a look at the following snippet where we select the features using default
parameters in SelectFromModel.
'''
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
# fetch a regression dataset
# in diabetes data we predict diabetes progression
# after one year based on some features
data = load_diabetes()
X = data["data"]
col_names = data["feature_names"]
y = data["target"]
# initialize the model
model = RandomForestRegressor()
# select from the model
sfm = SelectFromModel(estimator=model)
X_transformed = sfm.fit_transform(X, y)
# see which features were selected
support = sfm.get_support()
# get feature names
# get feature names
print([
x for x, y in zip(col_names, support) if y == True
])

''' One more thing that we are missing here is feature
selection using models that have L1 (Lasso) penalization. When we have L1
penalization for regularization, most coefficients will be 0 (or close to 0), and we
select the features with non-zero coefficients. You can do it by just replacing
random forest in the snippet of selection from a model with a model that supports
L1 penalty, e.g. lasso regression. All tree-based models provide feature importance
so all the model-based snippets shown in this chapter can be used for XGBoost,
LightGBM or CatBoost'''