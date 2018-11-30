from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

from models.model import model

import sys, os, pdb

import numpy as np
import utils.dgm as dgm 

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


""" 
Implementation of semi-supervised DGMs from Kingma et al. (2014): p(z) * p(y) * p(x|y,z) 
Inference network: q(z,y|x) = q(y|x) * q(z|y,x) 
"""

class m2(model):
   
    def __init__(self, n_x, n_y, n_z=2, n_hid=[4], alpha=0.1, x_dist='Gaussian', nonlinearity=tf.nn.relu, batchnorm=False, l2_reg=0.3, mc_samples=1,ckpt=None):
	""" TODO: Add any parameters necessary for initialization of the model """	
	super(m2, self).__init__(n_x, n_y, n_z, n_hid, x_dist, nonlinearity, batchnorm, mc_samples, alpha, l2_reg, ckpt)
	""" TODO: add any general terms we want to have here """

    def build_model(self):
	self.create_placeholders()
	self.initialize_networks()
	
	""" TODO: Define model components and variables """
	self.predictions = self.predict(self.x)

    def compute_loss(self):
	""" TODO: define model loss """
	pass 

    def labeled_loss(self, x, y):
	""" TODO: define labeled loss computation """
	pass

    def unlabeled_loss(self, x):
	""" TODO: define unlabeled loss computations """
	pass

    def qy_loss(self, x, y):
	""" TODO: define additional loss penalties """
	pass

    def sample_z(self, x, y):
	""" TODO: define sampling procedure for z """
	pass

    def compute_logpx(self, x, y, z):
	""" TODO: define input for x sampling """
	px_in = something
	if self.x_dist == 'Gaussian':
            mean, log_var = dgm.forwardPassGauss(px_in, self.px_yz, self.n_hid, self.nonlinearity, self.bn, scope='px_yz')
            return dgm.gaussianLogDensity(x, mean, log_var)
        elif self.x_dist == 'Bernoulli':
            logits = dgm.forwardPassCatLogits(px_in, self.px_yz, self.n_hid, self.nonlinearity, self.bn, scope='px_yz')
            return dgm.bernoulliLogDensity(x, logits)

    def predict(self, x):
	""" TODO: define prediction procedure for new inputs """
	pass
 
    def encode(self, x, y=None, n_iters=100):
	""" TODO: define procedure for encoding x into z """
	return z

    def compute_acc(self, x, y):
	y_ = self.predict(x)
	acc =  tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_,axis=1), tf.argmax(y, axis=1)), tf.float32))
	return acc 

    def initialize_networks(self):
	""" TODO: initialize necessary networks for model """
	pass

    def print_verbose1(self, epoch, fd, sess):
	total, elbo_l, elbo_u = sess.run([self.compute_loss(), self.elbo_l, self.elbo_u] ,fd)
	train_acc, test_acc = sess.run([self.train_acc, self.test_acc], fd)	
	print("Epoch: {}: Total: {:5.3f}, Labeled: {:5.3f}, Unlabeled: {:5.3f}, Training: {:5.3f}, Testing: {:5.3f}".format(epoch, total, elbo_l, elbo_u, train_acc, test_acc))	

    def print_verbose2(self, epoch, fd, sess):
	self.phase = False
	zm_test, zlv_test, z_test = self.sample_(self.x_test,self.y_test)
        zm_train, zlv_train, z_train = self.sample_z(self.x_train,self.y_train)
        lpx_test, lpx_train, acc_train, acc_test = sess.run([self.compute_logpx(self.x_test, z_test, self.y_test),
                                                                  self.compute_logpx(self.x_train, z_train, self.y_train),
                                                                  self.train_acc, self.test_acc], feed_dict=fd)
	print('Epoch: {}, logpx: {:5.3f}, Train: {:5.3f}, Test: {:5.3f}'.format(epoch, np.mean(lpx_train), np.mean(klz_train), acc_train, acc_test ))

    def _unlabeled_loss(self, x):
        """ Compute necessary terms for unlabeled loss (per data point) """
        weights = self.predict(x) 
        EL_l = 0 
        for label in range(self.n_y):
            y = self.generate_class(label, tf.shape(self.x_u)[0])
            EL_l += tf.multiply(weights[:,label], self.labeled_loss(x, y))
        ent_qy = -tf.reduce_sum(tf.multiply(weights, tf.log(weights+1e-10)), axis=1)
        return EL_l + ent_qy

    def generate_class(self, k, num):
        """ create one-hot encoding of class k with length num """
	y_u = tf.reshape(tf.tile(tf.eye(self.n_y), [1, num]), [-1, self.n_y])
	return y_u[num*k:num*(k+1)]
	
