'''The simplest form of selecting features would be to remove features with very
low variance. If the features have a very low variance (i.e. very close to 0), they
are close to being constant and thus, do not add any value to any model at all. It
would just be nice to get rid of them and hence lower the complexity. Please note
that the variance also depends on scaling of the data. Scikit-learn has an
implementation for VarianceThreshold that does precisely this.
'''
from sklearn.feature_selection import VarianceThreshold
data = ...
var_thresh = VarianceThreshold(threshold=0.1)
transformed_data = var_thresh.fit_transform(data)
# transformed data will have all columns with variance less
# than 0.1 removed