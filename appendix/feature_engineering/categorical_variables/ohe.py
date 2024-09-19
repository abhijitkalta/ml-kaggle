'''
Use mostly with linear models, svm, neural networks
'''

import numpy as np
from sklearn import preprocessing
# create random 1-d array with 1001 different categories (int)
example = np.random.randint(1000, size=1000000)
# initialize OneHotEncoder from scikit-learn
# keep sparse = False to get dense array
ohe = preprocessing.OneHotEncoder(sparse=False)
# fit and transform data with dense one hot encoder
ohe_example = ohe.fit_transform(example.reshape(-1, 1))
# print size in bytes for dense array
print(f"Size of dense array: {ohe_example.nbytes}")
# initialize OneHotEncoder from scikit-learn
# keep sparse = True to get sparse array
ohe = preprocessing.OneHotEncoder(sparse=True)
# fit and transform data with sparse one-hot encoder
ohe_example = ohe.fit_transform(example.reshape(-1, 1))
# print size of this sparse matrix
print(f"Size of sparse array: {ohe_example.data.nbytes}")
full_size = (
ohe_example.data.nbytes +
ohe_example.indptr.nbytes + ohe_example.indices.nbytes
)
# print full size of this sparse matrix
print(f"Full size of sparse array: {full_size}")

'''
We can also check this using a simple python snippet.
═════════════════════════════════════════════════════════════════════════
import numpy as np
# create our example feature matrix
example = np.array(
[
[0, 0, 1],
[1, 0, 0],
[1, 0, 1]
]
)
# print size in bytes
print(example.nbytes)
═════════════════════════════════════════════════════════════════════════
This code will print 72 as we calculated before. But do we need to store all the
elements of this matrix? No. As mentioned before we are only interested in 1s. 0s
are not that important because anything multiplied with 0 will be zero and 0
added/subtracted to/from anything doesn’t make any difference. One way to
represent this matrix only with ones would be some kind of dictionary method in
which keys are indices of rows and columns and value is 1:
═════════════════════════════════════════════════════════════════════════
(0, 2) 1
(1, 0) 1
(2, 0) 1
(2, 2) 1
══════════════════

Above is the sparse matrix

Suppose we represent each category of the ord_2 variable by a vector. This vector
is of the same size as the number of categories in the ord_2 variable. In this specific
case, each vector is of size six and has all zeros except at one position. Let’s look
at this particular table of vectors.
Freezing 0 0 0 0 0 1
Warm 0 0 0 0 1 0
Cold 0 0 0 1 0 0
Boiling Hot 0 0 1 0 0 0
Hot 0 1 0 0 0 0
Lava Hot 1 0 0 0 0 0
We see that the size of vectors is 1x6, i.e. there are six elements in the vector. Where
does this number come from? If you look carefully, you will see that there are six
categories, as mentioned before. When one-hot encoding, the vector size has to be
same as the number of categories we are looking at. Each vector has a 1 and rest all
other values are 0s.

'''