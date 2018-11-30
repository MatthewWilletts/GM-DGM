from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.distributions import RelaxedOneHotCategorical as Gumbel
from utils.utils import softmax
""" Module containing shared functions and structures for DGMS """

glorotNormal = xavier_initializer(uniform=False)
initNormal = tf.random_normal_initializer(stddev=1e-3)

"""  Probability functions """


def gaussianLogDensity(inputs, mu, log_var):
    """ Gaussian log density """
    D = tf.cast(tf.shape(inputs)[-1], tf.float32)
    xc = inputs - mu
    return -0.5 * (tf.reduce_sum((xc * xc) / tf.exp(log_var), axis=-1) +
                   tf.reduce_sum(log_var, axis=-1) + D * tf.log(2.0 * np.pi))


def gaussianLogDensity_axis(inputs, mu, log_var):
    """ Gaussian log density, but with no summing along axes """
    xc = inputs - mu
    return -0.5 * ((xc * xc) / tf.exp(log_var) + log_var + tf.log(2.0 * np.pi))


def gaussianLogDensity_np(inputs, mu, log_var):
    """ Gaussian log density, using numpy """
    D = inputs.shape[-1]
    xc = inputs - mu
    return -0.5 * (np.sum((xc * xc) / np.exp(log_var), axis=-1) +
                   np.sum(log_var, axis=-1) + D * np.log(2.0 * np.pi))


def standardNormalLogDensity(inputs):
    """ Standard normal log density """
    mu = tf.zeros_like(inputs)
    log_var = tf.log(tf.ones_like(inputs))
    return gaussianLogDensity(inputs, mu, log_var)


def bernoulliLogDensity(inputs, logits):
    """ Bernoulli log density """
    return -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=inputs, logits=logits), axis=-1)


def multinoulliLogDensity(inputs, preds, on_priors=False):
    """ Categorical log density """
    if on_priors is False:
        return -tf.nn.softmax_cross_entropy_with_logits(
            labels=inputs, logits=preds)
    else:
        return tf.reduce_sum(inputs * tf.log(preds + 1e-10), axis=-1)


def multinoulliUniformLogDensity(logits, dim=-1, order=True,):
    """ Uniform Categorical log density """
    if order is True:
        labels = tf.divide(tf.ones_like(logits),
                           tf.norm(tf.ones_like(logits),
                                   ord=1, axis=dim,
                                   keepdims=True))
        return - tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                         logits=logits,
                                                         dim=dim)
    else:
        labels = tf.ones_like(logits)
        return -tf.nn.softmax_cross_entropy_with_logits(labels=logits,
                                                        logits=labels,
                                                        dim=dim)


def discreteUniformKL(logits, n_size, dim=-1):
    """ KL divergence for discrete/categorical probabilties

    returns KL(q||p) where q is a tf tensor in logits and
    p, not given, is uniform.
    """
    return - tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=tf.nn.softmax(logits),
        logits=logits,
        dim=dim) + tf.log(n_size)


def discreteKL(q_logits, p_true, n_size, dim=-1):
    """ KL divergence for discrete/categorical probabilties

    returns KL(q||p) where q is a tf tensor in logits and p is given a p_true
    is a probability vector.
    """

    q_prob = tf.nn.softmax(q_logits)
    return - tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=q_prob, logits=q_logits, dim=dim) - tf.reduce_sum(
        q_prob * tf.log(p_true + 1e-10), axis=dim)


def discreteUniformKL_np(logits, n_size, dim=-1):
    """ KL divergence for discrete/categorical probabilties

    returns KL(q||p) where q is a np array in logits and
    p, not given, is uniform.
    """
    return - softmax(logits) + np.log(n_size)


def discreteUniformKL_np_probs(probs, n_size, dim=-1):
    """ KL divergence for discrete/categorical probabilties

    returns KL(q||p) where q is a np array of probabilities and
    p, not given, is uniform.
    """
    return np.sum(probs * np.log(probs + 10e-10), axis=dim) + np.log(n_size)


def gumbelLogDensity(inputs, logits, temp):
    """ log density of a Gumbel distribution for tf inputs"""
    dist = Gumbel(temperature=temp, logits=logits)
    return dist.log_prob(inputs)


def sampleNormal(mu, logvar, mc_samps):
    """ return a reparameterized sample from a Gaussian distribution """
    shape = tf.concat([tf.constant([mc_samps]), tf.shape(mu)], axis=-1)
    eps = tf.random_normal(shape, dtype=tf.float32)
    return mu + eps * tf.sqrt(tf.exp(logvar))


def sampleNormal_np(mu, logvar, mc_samps):
    """ return a reparameterized sample from a Gaussian distribution """
    eps = np.random.normal(size=mu.shape[0])
    return np.transpose(mu.T + np.sqrt(np.exp(logvar)).T * eps)


def sampleGumbel(logits, temp):
    """ return a reparameterized sample from a Gaussian distribution """
    shape = tf.shape(logits)
    U = tf.random_uniform(shape, minval=0, maxval=1)
    eps = -tf.log(-tf.log(U + 1e-10) + 1e-10)
    y = logits + eps
    return tf.nn.softmax(y / temp)


def standardNormalKL(mu, logvar):
    """ compute the KL divergence between a Gaussian and standard normal """
    return -0.5 * tf.reduce_sum(1 + logvar - mu**2 - tf.exp(logvar), axis=-1)


def gaussianKL(mu1, logvar1, mu2, logvar2):
    """ compute the KL divergence between two arbitrary Gaussians """
    return -0.5 * (1 + logvar1 - logvar2 - tf.exp(logvar1) / tf.exp(logvar2) -
                   ((mu1 - mu2)**2) / tf.exp(logvar2))


def gaussianKL_np(mu1, logvar1, mu2, logvar2):
    """ compute the KL divergence between two arbitrary Gaussians """
    return -0.5 * np.sum(1 + logvar1 - logvar2 - np.exp(logvar1) /
                         np.exp(logvar2) - ((mu1 - mu2)**2) / np.exp(logvar2),
                         axis=-1)


"""Neural Network modules """


def forwardPass(network, input_var):
    return network(input_var)


def forwardPassList(list_of_networks, input_var):
    return [forwardPass(network, input_var) for network in list_of_networks]
