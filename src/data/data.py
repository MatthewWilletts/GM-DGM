from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os, pickle, gzip, pdb
if sys.version_info[0]<3:
    import cPickle
import numpy as np
import pandas
from data.mnist import mnist
from data.SSL_DATA import SSL_DATA
from keras.datasets import cifar10, cifar100
import scipy.io
from keras.utils import np_utils
import utils.preprocessing as pp
from scipy.stats import mode
import datetime

""" Class for data to handle loading and arranging """


def create_semisupervised(x_train, y_train, num_classes, num_labels, seed):
    np.random.seed(seed)
    if type(num_labels) is not list:
        num_labels = [int(float(num_labels) / num_classes)] * num_classes
    x_u, y_u, x_l, y_l = [], [], [], []
    for c in range(num_classes):
        indices = np.where(y_train[:, c] == 1)
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
            y_u.append(ycls)
    x_labelled, y_labelled = np.concatenate(x_l), np.concatenate(y_l)
    x_unlabelled, y_unlabelled = np.concatenate(x_u), np.concatenate(y_u)
    return x_labelled, y_labelled, x_unlabelled, y_unlabelled


def encode_onehot(labels):
    n, d = labels.shape[0], np.max(labels) + 1
    return np.eye(d)[labels]


def rgb2gray(images):
    """Convert images from rbg to grayscale
    """
    return np.expand_dims(np.dot(images, [0.2989, 0.5870, 0.1140]), axis=3)


def load_dataset(dataset_name, preproc=True, threshold=0.1, greyscale=False):
    """ load up data, either mnist, cifar10, cifar100 or svhn
    from mnist, optional arguments:
    'threshold' keeps only elements of data where over the dataset their
     variance > threshold. This is to prevent perfect matching to
     pixels in e.g. the corners of the image that are always =0

    for svhn, optional arguments:
    'preproc' to do optional PCA extraction of data
    'greyscale' to convert to greyscale images
    """

    if dataset_name == 'mnist':
        target = '/jmain01/home/JAD017/sjr01/mxw35-sjr01/Projects/generativeSSL-master/data/mnist.pkl.gz'
        data = mnist(target, threshold=threshold)

        x_train, y_train = data.x_train, data.y_train
        x_test, y_test = data.x_test, data.y_test

        binarize = True
        x_dist = 'Bernoulli'
        n_y = 10
        f_enc, f_dec = lambda x: x, lambda x: x
        n_x = x_train.shape[1]

    elif dataset_name == 'cifar10':

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        binarize = False
        x_dist = 'Gaussian'
        n_y = 10
        f_enc, f_dec = lambda x: x, lambda x: x
        n_x = x_train.shape[1]

    elif dataset_name == 'cifar100':
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()

        binarize = False
        x_dist = 'Gaussian'
        n_y = 100
        f_enc, f_dec = lambda x: x, lambda x: x
        n_x = x_train.shape[1]

    elif dataset_name == 'svhn':

        train = scipy.io.loadmat('/jmain01/home/JAD017/sjr01/mxw35-sjr01/SVHN/train_32x32.mat')
        train_x = train['X'].swapaxes(0, 1).T.reshape((train['X'].shape[3], -1)).T
        train_y = train['y'].reshape((-1)) - 1

        test = scipy.io.loadmat('/jmain01/home/JAD017/sjr01/mxw35-sjr01/SVHN/test_32x32.mat')
        test_x = test['X'].swapaxes(0, 1).T.reshape((test['X'].shape[3], -1)).T
        test_y = test['y'].reshape((-1)) - 1

        extra = scipy.io.loadmat('/jmain01/home/JAD017/sjr01/mxw35-sjr01/SVHN/extra_32x32.mat')
        extra_x = extra['X'].swapaxes(0, 1).T.reshape((extra['X'].shape[3], -1)).T
        extra_y = extra['y'].reshape((-1)) - 1

        train_x = train_x.astype(np.float32) / 256.
        test_x = test_x.astype(np.float32) / 256.
        extra_x = extra_x.astype(np.float32) / 256.

        y_train = np_utils.to_categorical(train_y)
        y_test = np_utils.to_categorical(test_y)
        y_extra = np_utils.to_categorical(extra_y)

        x_train = np.hstack((train_x, extra_x))
        y_train = np.vstack((y_train, y_extra))

        if greyscale is False:
            binarize = False
            x_dist = 'Gaussian'
        elif greyscale is True:
            x_train = rgb2gray(x_train).astype(np.float32)
            test_x = rgb2gray(test_x).astype(np.float32)

        n_y = 10

        if preproc is True:
            f_enc, f_dec, pca_params = pp.PCA(train_x[:, :10000],
                                              cutoff=1000, toFloat=False)
            x_train = f_enc(x_train).astype(np.float32).T
            x_test = f_enc(test_x).astype(np.float32).T
        else:
            x_train = np.transpose(x_train)
            x_test = np.transpose(test_x)
            f_enc, f_dec = lambda x: x, lambda x: x

        n_x = x_train.shape[1]

    return x_train, y_train, x_test, y_test, binarize, x_dist, n_y, n_x, f_enc, f_dec


def make_dataset(learning_paradigm, dataset_name, x_test, y_test, x_train=None,
                 y_train=None, num_labelled=None, seed=0,
                 number_of_classes_to_add=None, do_split=True,
                 x_labelled=None, y_labelled=None,
                 x_unlabelled=None, y_unlabelled=None):
    """ turn data into a SSL_DATA instance

    learning paradigms: 'string' one of: supervised, semisupervised,
                                semi-unsupervised and unsupervised
    dataset_name: 'string' for internal name of returned SSL_DATA instance
    x_test, y_test: 'np array's of test data, with y_test one-hot
    x_train, y_train: 'np array's of training data, to be given if do_split
        is True.
    num_labelled: 'int' or 'list': only used if do_split is true. If int, then
        it is the number of labelled examples for each class. If a list, then
        gives the number of labelled examples for each class, which now can be
        different.
    seed: 'int' seed for creating split data
    number_of_classes_to_add: 'int', default 'None': the number of extra empty
        classes to add. Used when doing semi-unsupervised or unsupervised
        learning.
    do_spit: 'bool', if True then make semisupervised or semi-unsupervised data
        by extracting out the num_labelled examples for each class
    x_labelled, y_labelled, x_unlabelled, y_unlabelled: 'np array',
        these are to be used for pre-split data for semi-unsupervised or
        unsupervised learning, where we just pack them into SSL_DATA
    """


    if learning_paradigm == 'unsupervised' and number_of_classes_to_add is not None:
        y_train = np.concatenate((y_train,
                                  np.zeros((y_train.shape[0],
                                            number_of_classes_to_add))),
                                 axis=1)
        y_test = np.concatenate((y_test,
                                 np.zeros((y_test.shape[0],
                                           number_of_classes_to_add))),
                                axis=1)

    if learning_paradigm == 'unsupervised' or learning_paradigm == 'supervised':
        Data = SSL_DATA(x_train, y_train, x_test=x_test, y_test=y_test,
                        x_labelled=x_train, y_labelled=y_train,
                        dataset=dataset_name)

    if learning_paradigm == 'semisupervised' or learning_paradigm == 'semi-unsupervised':
        print('here')
        if do_split is True:
            x_labelled, y_labelled, x_unlabelled, y_unlabelled =\
                create_semisupervised(x_train, y_train,
                                      num_classes=y_train.shape[1],
                                      num_labels=num_labelled, seed=seed)
        if number_of_classes_to_add is not None:
            y_unlabelled = np.concatenate((y_unlabelled,
                                          np.zeros((y_unlabelled.shape[0],
                                                    number_of_classes_to_add))),
                                          axis=1)
            y_labelled = np.concatenate((y_labelled,
                                        np.zeros((y_labelled.shape[0],
                                                  number_of_classes_to_add))),
                                        axis=1)
            y_test = np.concatenate((y_test,
                                    np.zeros((y_test.shape[0],
                                              number_of_classes_to_add))),
                                    axis=1)

        Data = SSL_DATA(x_unlabelled, y_unlabelled, x_test=x_test, y_test=y_test,
                x_labelled=x_labelled, y_labelled=y_labelled, dataset=dataset_name)
    return Data


def from_conf_to_preds(conf_mat):
    """ reverses the creation of a confusion matrix, giving us back lists for
    ground truth and predictions. Obviously doesnt preserved orgininal data
    order

    conf_mat 'np array': confusion matrix in format of sklearn confusion matrix
    """
    y_true = list()
    y_pred = list()
    # i will go over rows
    # j over columns
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            y_true.append(i * np.ones(conf_mat[i, j]))
            y_pred.append(j * np.ones(conf_mat[i, j]))
    #
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return y_true, y_pred


def find_relabelled_y_pred(y_true, y_pred):
    """ Peforms cluster-and-label approach on predicitons, assigning predicted
    classes to their most prevalent true class

    """
    real_pred = np.zeros_like(y_pred)
    for cat in range(int(np.max(y_pred)) + 1):
        idx = y_pred == cat
        lab = y_true[idx]
        if len(lab) == 0:
            continue
        real_pred[y_pred == cat] = mode(lab).mode[0]
    return real_pred
