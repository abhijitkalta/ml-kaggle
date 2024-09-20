# Either fill them with 0 or better to fill them with mean or knn values
import numpy as np
from sklearn import impute
# create a random numpy array with 10 samples
# and 6 features and values ranging from 1 to 15
X = np.random.randint(1, 15, (10, 6))
# convert the array to float
X = X.astype(float)
# randomly assign 10 elements to NaN (missing)
X.ravel()[np.random.choice(X.size, 10, replace=False)] = np.nan
# use 3 nearest neighbours to fill na values
knn_imputer = impute.KNNImputer(n_neighbors=2)
knn_imputer.fit_transform(X)

''' Always remember that imputing values for tree-based models is unnecessary as they
 can handle it themselves.
 And always remember to scale or normalize your
features if you are using linear models like logistic regression or a model like SVM.
Tree-based models will always work fine without any normalization of features.
'''