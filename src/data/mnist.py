from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
import gzip
import numpy as np
if sys.version_info[0] < 3:
    import cPickle


""" Class for mnist data to handle data loading and arranging """


class mnist:

    def __init__(self, path='data/mnist.pkl.gz', threshold=0.1):
        with gzip.open(path, 'rb') as f:
            train_set, val_set, test_set = cPickle.load(f)
            self.x_train = train_set[0]
            self.y_train = self.encode_onehot(train_set[1])
            if not len(val_set[0]) is 0:
                self.x_val = val_set[0]
                self.y_val = self.encode_onehot(val_set[1])
                self.n_val = self.x_val.shape[0]
            self.x_test, self.y_test = test_set[0], self.encode_onehot(test_set[1])
            self.n_train, self.n_test = self.x_train.shape[0], self.x_test.shape[0]
            self.drop_dimensions(threshold)
            self.x_dim, self.num_classes = self.x_train.shape[1], self.y_train.shape[1]

    def drop_dimensions(self, threshold=0.1):
        stds = np.std(self.x_train, axis=0)
        good_dims = np.where(stds > threshold)[0]
        self.x_train = self.x_train[:, good_dims]
        if hasattr(self, 'x_val'):
            self.x_val = self.x_val[:, good_dims]
        self.x_test = self.x_test[:, good_dims]

    def encode_onehot(self, labels):
        d = np.max(labels) + 1
        return np.eye(d)[labels]
