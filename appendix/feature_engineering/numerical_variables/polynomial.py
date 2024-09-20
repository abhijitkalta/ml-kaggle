import numpy as np
import pandas as pd

# generate a random dataframe with
# 2 columns and 100 rows
df = pd.DataFrame(
np.random.rand(100, 2),
columns=[f"f_{i}" for i in range(1, 3)]
)

from sklearn import preprocessing
# initialize polynomial features class object
# for two-degree polynomial features
pf = preprocessing.PolynomialFeatures(
degree=2,
interaction_only=False,
include_bias=False
)
# fit to the features
pf.fit(df)
# create polynomial features
poly_feats = pf.transform(df)
# create a dataframe with all the features
num_feats = poly_feats.shape[1]
df_transformed = pd.DataFrame(
poly_feats,
columns=[f"f_{i}" for i in range(1, num_feats + 1)]
)