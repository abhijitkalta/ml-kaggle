import pandas as pd
from sklearn import preprocessing
# read the data
df = pd.read_csv("../input/cat_train.csv")
# fill NaN values in ord_2 column
df.loc[:, "ord_2"] = df.ord_2.fillna("NONE")
# initialize LabelEncoder
lbl_enc = preprocessing.LabelEncoder()
# fit label encoder and transform values on ord_2 column
# P.S: do not use this directly. fit first, then transform
df.loc[:, "ord_2"] = lbl_enc.fit_transform(df.ord_2.values)

'''
Label encoder doesn't handle NaN values.
Label encoding is a simple way of encoding categorical values to create
a dictionary that maps these values to numbers starting from 0 to N-1, where N is
the total number of categories in a given feature.

mapping = {
"Freezing": 0,
"Warm": 1,
"Cold": 2,
"Boiling Hot": 3,
"Hot": 4,
"Lava Hot": 5
}
═════════════════════════════════════════════════════════════════════════
Now, we can read the dataset and convert these categories to numbers easily.
═════════════════════════════════════════════════════════════════════════
import pandas as pd
df = pd.read_csv("../input/cat_train.csv")
df.loc[:, "ord_2"] = df.ord_2.map(mapping)
'''