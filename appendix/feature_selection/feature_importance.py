'''
We saw two different greedy ways to select features from a model. But you can also
fit the model to the data and select features from the model by the feature
coefficients or the importance of features. If you use coefficients, you can select
a threshold, and if the coefficient is above that threshold, you can keep the feature
else eliminate it.
Let’s see how we can get feature importance from a model like random forest.
'''
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
# fetch a regression dataset
# in diabetes data we predict diabetes progression
# after one year based on some features
data = load_diabetes()
X = data["data"]
col_names = data["feature_names"]
y = data["target"]
# initialize the model
model = RandomForestRegressor()
# fit the model
model.fit(X, y)

importances = model.feature_importances_
idxs = np.argsort(importances)
plt.title('Feature Importances')
plt.barh(range(len(idxs)), importances[idxs], align='center')
plt.yticks(range(len(idxs)), [col_names[i] for i in idxs])
plt.xlabel('Random Forest Feature Importance')
plt.show()