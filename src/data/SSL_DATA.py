from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

""" Generate a data structure to support SSL models. Expects:
x - np array: N rows, d columns
y - np array: N rows, k columns (one-hot encoding)
"""


class SSL_DATA:
    """ Class for appropriate data structures """
    def __init__(self, x, y, x_test=None, y_test=None, x_labelled=None,
                 y_labelled=None, train_proportion=0.7, labelled_proportion=0.3,
                 dataset='', seed=None):
        self.INPUT_DIM = x.shape[1]
        self.NUM_CLASSES = y.shape[1]
        self.NAME = dataset

        if x_test is None:
            self.N = x.shape[0]
            self.TRAIN_SIZE = int(np.round(train_proportion * self.N))
            self.TEST_SIZE = int(self.N - self.TRAIN_SIZE)
        else:
            self.TRAIN_SIZE = x.shape[0]
            self.TEST_SIZE = x_test.shape[0]
            self.N = self.TRAIN_SIZE + self.TEST_SIZE

        # create necessary data splits
        if seed:
            np.random.seed(seed)
        if x_test is None:
            xtrain, ytrain, xtest, ytest = self._split_data(x, y)
        else:
            xtrain, ytrain, xtest, ytest = x, y, x_test, y_test
        if x_labelled is None:
            self.NUM_LABELLED = int(np.round(self.TRAIN_SIZE * labelled_proportion))
            self.NUM_UNLABELLED = int(self.TRAIN_SIZE - self.NUM_LABELLED)
            x_labelled, y_labelled, x_unlabelled, y_unlabelled = \
                self._create_semisupervised(xtrain, ytrain)
        else:
            x_unlabelled, y_unlabelled = xtrain, ytrain

        self.NUM_LABELLED = x_labelled.shape[0]
        self.NUM_UNLABELLED = x_unlabelled.shape[0]

        # create appropriate data dictionaries
        self.data = {}
        self.data['x_train'] = np.concatenate((x_labelled, x_unlabelled))
        self.data['y_train'] = np.concatenate((y_labelled, y_unlabelled))
        self.TRAIN_SIZE = self.data['x_train'].shape[0]
        self.data['x_u'], self.data['y_u'] = x_unlabelled, y_unlabelled
        self.data['x_l'], self.data['y_l'] = x_labelled, y_labelled
        self.data['x_test'], self.data['y_test'] = xtest, ytest

        # counters and indices for minibatching
        self._start_labelled, self._start_unlabelled = 0, 0
        self._epochs_labelled = 0
        self._epochs_unlabelled = 0
        self._start_regular = 0
        self._epochs_regular = 0

    def _split_data(self, x, y):
        """ split the data according to the proportions """
        indices = range(self.N)
        np.random.shuffle(indices)
        train_idx = indices[:self.TRAIN_SIZE]
        test_idx = indices[self.TRAIN_SIZE:]
        return (x[train_idx, :], y[train_idx, :], x[test_idx, :], y[test_idx, :])

    def _create_semisupervised(self, x, y):
        """ split training data into labelled and unlabelled """
        indices = range(self.TRAIN_SIZE)
        np.random.shuffle(indices)
        l_idx, u_idx = indices[:self.NUM_LABELLED], indices[self.NUM_LABELLED:]
        return (x[l_idx, :], y[l_idx, :], x[u_idx, :], y[u_idx, :])

    def create_semisupervised(self, x_train, y_train, num_classes,
                              num_labels, seed):
        np.random.seed(seed)
        if type(num_labels) is not list:
            num_labels = [num_labels] * num_classes
        x_u, y_u, x_l, y_l = [], [], [], []
        for c in range(num_classes):
            indices = np.where(y_train[:,c] == 1)
            xcls, ycls = x_train[indices], y_train[indices]
            perm = np.random.permutation(xcls.shape[0])
            xcls = xcls[perm]
            ycls = ycls[perm]
            if num_labels[c] > 0:
                x_l.append(xcls[:num_labels[c]])
                y_l.append(ycls[:num_labels[c]])
                x_u.append(xcls[num_labels[c]:])
                y_u.append(ycls[num_labels[c]:])
            else:
                x_u.append(xcls)
                y_u.append(ycls[num_labels[c]:])
        x_labelled, y_labelled = np.concatenate(x_l), np.concatenate(y_l)
        x_unlabelled, y_unlabelled = np.concatenate(x_u), np.concatenate(y_u)
        return x_labelled, y_labelled, x_unlabelled, y_unlabelled

    def recreate_semisupervised(self, seed):
        self.data['x_l'], self.data['y_l'], self.data['x_u'], self.data['y_u'] =\
            self.create_semisupervised(self.data['x_train'],
                                       self.data['y_train'],
                                       num_classes=self.NUM_CLASSES,
                                       num_labels=list(np.sum(self.data['y_l'],
                                                       axis=0, dtype=int)),
                                       seed=seed)

    def next_batch(self, labelled_BATCHSIZE, UNlabelled_BATCHSIZE):
        x_l_batch, y_l_batch = self.next_batch_labelled(labelled_BATCHSIZE)
        x_u_batch, y_u_batch = self.next_batch_unlabelled(UNlabelled_BATCHSIZE)
        return (x_l_batch, y_l_batch, x_u_batch, y_u_batch)

    def next_batch_labelled(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._start_labelled
        # Shuffle for the first epoch
        if self._epochs_labelled == 0 and start == 0 and shuffle:
            perm0 = np.arange(self.NUM_LABELLED)
            np.random.shuffle(perm0)
            self.data['x_l'] = self.data['x_l'][perm0, :]
            self.data['y_l'] = self.data['y_l'][perm0, :]
            # Go to the next epoch
        if start + batch_size > self.NUM_LABELLED:
            # Finished epoch
            self._epochs_labelled += 1
            # Get the rest examples in this epoch
            rest_num_examples = self.NUM_LABELLED - start
            inputs_rest_part = self.data['x_l'][start:self.NUM_LABELLED]
            labels_rest_part = self.data['y_l'][start:self.NUM_LABELLED]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self.NUM_LABELLED)
                np.random.shuffle(perm)
                self.data['x_l'] = self.data['x_l'][perm]
                self.data['y_l'] = self.data['y_l'][perm]
            # Start next epoch
            start = 0
            self._start_labelled = batch_size - rest_num_examples
            end = self._start_labelled
            inputs_new_part = self.data['x_l'][start:end]
            labels_new_part = self.data['y_l'][start:end]
            return np.concatenate((inputs_rest_part, inputs_new_part), axis=0), np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._start_labelled += batch_size
            end = self._start_labelled
            return self.data['x_l'][start:end], self.data['y_l'][start:end]

    def next_batch_unlabelled(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._start_unlabelled
        # Shuffle for the first epoch
        if self._epochs_unlabelled == 0 and start == 0 and shuffle:
            perm0 = np.arange(self.NUM_UNLABELLED)
            np.random.shuffle(perm0)
            self.data['x_u'] = self.data['x_u'][perm0, :]
            self.data['y_u'] = self.data['y_u'][perm0, :]
        # Go to the next epoch
        if start + batch_size > self.NUM_UNLABELLED:
            # Finished epoch
            self._epochs_unlabelled += 1
            # Get the rest examples in this epoch
            rest_num_examples = self.NUM_UNLABELLED - start
            inputs_rest_part = self.data['x_u'][start:self.NUM_UNLABELLED, :]
            labels_rest_part = self.data['y_u'][start:self.NUM_UNLABELLED, :]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self.NUM_UNLABELLED)
                np.random.shuffle(perm)
                self.data['x_u'] = self.data['x_u'][perm]
                self.data['y_u'] = self.data['y_u'][perm]
            # Start next epoch
            start = 0
            self._start_unlabelled = batch_size - rest_num_examples
            end = self._start_unlabelled
            inputs_new_part = self.data['x_u'][start:end, :]
            labels_new_part = self.data['y_u'][start:end, :]
            return np.concatenate((inputs_rest_part, inputs_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._start_unlabelled += batch_size
            end = self._start_unlabelled
            return self.data['x_u'][start:end, :], self.data['y_u'][start:end, :]

    def next_batch_regular(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._start_regular
        # Shuffle for the first epoch
        if self._epochs_regular == 0 and start == 0 and shuffle:
            perm0 = np.arange(self.TRAIN_SIZE)
            np.random.shuffle(perm0)
            self.data['x_train'] = self.data['x_train'][perm0, :]
            self.data['y_train'] = self.data['y_train'][perm0, :]
        # Go to the next epoch
        if start + batch_size > self.TRAIN_SIZE:
            # Finished epoch
            self._epochs_regular += 1
            # Get the rest examples in this epoch
            rest_num_examples = self.TRAIN_SIZE - start
            inputs_rest_part = self.data['x_train'][start:self.TRAIN_SIZE, :]
            labels_rest_part = self.data['y_train'][start:self.TRAIN_SIZE, :]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self.TRAIN_SIZE)
                np.random.shuffle(perm)
                self.data['x_train'] = self.data['x_train'][perm]
                self.data['y_train'] = self.data['y_train'][perm]
            # Start next epoch
            start = 0
            self._start_regular = batch_size - rest_num_examples
            end = self._start_regular
            inputs_new_part = self.data['x_train'][start:end, :]
            labels_new_part = self.data['y_train'][start:end, :]
            return np.concatenate((inputs_rest_part, inputs_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._start_regular += batch_size
            end = self._start_regular
            return self.data['x_train'][start:end, :], self.data['y_train'][start:end, :]

    def next_batch_shuffle(self, batch_size):
        """ Return a random subsample of the data """
        perm0 = np.arange(self.TRAIN_SIZE)
        np.random.shuffle(perm0)
        self.data['x_train'] = self.data['x_train'][perm0, :]
        self.data['y_train'] = self.data['y_train'][perm0, :]
        return self.data['x_train'][:batch_size], self.data['y_train'][:batch_size]

    def sample_train(self, n_samples=1000):
        perm_train = np.arange(self.TRAIN_SIZE)
        np.random.shuffle(perm_train)
        self.data['x_train'] = self.data['x_train'][perm_train, :]
        self.data['y_train'] = self.data['y_train'][perm_train, :]
        return self.data['x_train'][:n_samples], self.data['y_train'][:n_samples]

    def sample_test(self, n_samples=1000):
        perm_test = np.arange(self.TEST_SIZE)
        np.random.shuffle(perm_test)
        return self.data['x_test'][perm_test[:n_samples]], self.data['y_test'][perm_test[:n_samples]]

    def query(self, idx):
        """ Implementation of an oracle for the data """
        self.data['x_l'] = np.vstack((self.data['x_l'], self.data['x_u'][idx]))
        self.data['y_l'] = np.vstack((self.data['y_l'], self.data['y_u'][idx]))
        self.data['x_u'] = np.delete(self.data['x_u'], [idx], axis=0)
        self.data['y_u'] = np.delete(self.data['y_u'], [idx], axis=0)
        self.NUM_LABELLED = self.data['x_l'].shape[0]
        self.NUM_UNLABELLED = self.data['x_u'].shape[0]

    def reset_counters(self):
        """ counters and indices for minibatching """
        self._start_labelled, self._start_unlabelled = 0, 0
        self._epochs_labelled = 0
        self._epochs_unlabelled = 0
        self._start_regular = 0
        self._epochs_regular = 0
