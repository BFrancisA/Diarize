import numpy as np
from keras.models import Sequential
from keras.layers import Embedding


model = Sequential()

# Here input_dim is the vocab size (total number of unique words, symbols, labels, integers, etc)
# that a single item in the sequence can have.
#
#
#                   input_dim, output_dim, input_length = length of the sequence
model.add(Embedding(1000,      64,         input_length=10))
# the model will take as input an integer matrix of size (batch, input_length).
# the largest integer (i.e. word index) in the input should be no larger than 999 (vocabulary size).
# now model.output_shape == (None, 10, 64), where None is the batch dimension.


"""
What this is really doing is finding a better way of doing one-hot encoding.
One-hot encoding a word where the vocab size is 1000 would result in a 1000-dimension vector.  One val is 1, the other 999 are 0.
This is super wasteful.  Embedding calculates a unique vector for each possible symbol (e.g. word) in the vocab.
Here in the example, 10 words are converted to 10 64-dim vectors.  This is way smaller than 10 1000-dim one-hot encoded vectors.
"""


# One 2-D array: 32 rows X 10 cols  Values are random in range [0 to 1000)
# 32 samples: each sample is a sequence of 10 integers.  These integers are in the range of 0 to 999 inclusive.
input_array = np.random.randint(1000, size=(32, 10))

model.compile('rmsprop', 'mse')

model.summary()

output_array = model.predict(input_array)

print(output_array.shape)

assert output_array.shape == (32, 10, 64)   # 32 matrices, each is 10 X 64