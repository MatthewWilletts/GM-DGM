from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.model import model

import numpy as np
import utils.dgm as dgm

import tensorflow as tf

from keras.layers import Dense, Activation
from keras.models import Sequential
from keras import initializers


""" 
Implementation of semi-supervised DGMs from Kingma et al.(2014):
p(x,y,z) = p(z) * p(y) * p(x|y,z) 
Inference network: q(z,y|x) = q(y|x) * q(z|y,x) 

Here we use keras layers to implement MLPs
"""


class m2(model):

    def __init__(self, n_x, n_y, n_z=2, x_dist='Gaussian', mc_samples=1,
                 alpha=0.1, l2_reg=0.3, ckpt=None,
                 learning_paradigm='supervised', name='m2', prior=None,
                 analytic_kl=False, output_dir=None):

        self.reg_term = tf.placeholder(tf.float32, shape=[], name='reg_term')
        if prior is None:
            self.prior = tf.constant(np.array([1.0 / n_y] * n_y),
                                     dtype=tf.float32, shape=[1, n_y],
                                     name='prior_p_y')
        else:
            self.prior = tf.constant(prior, dtype=tf.float32, shape=[1, n_y],
                                     name='prior_p_y')

        super(m2, self).__init__(n_x, n_y, n_z, x_dist, mc_samples, l2_reg,
                                 alpha, ckpt, learning_paradigm, name,
                                 analytic_kl, output_dir)
        """ TODO: add any general terms we want to have here """


    def build_model(self):
        """ Define model components and variables """
        self.create_placeholders()

        glorot_initializer = initializers.glorot_normal()
        normal_initializer = initializers.random_normal(stddev=0.001)

        ## Inference Networks:
        # Make q(y|x) network
        self.q_y_x_model = Sequential()
        self.q_y_x_model.add(Dense(self.intermediate_dim, name='hidden_1',
                                   kernel_initializer=glorot_initializer,
                                   bias_initializer=normal_initializer,
                                   input_dim=self.n_x))

        self.q_y_x_model.add(Activation('relu'))

        self.q_y_x_model.add(Dense(self.intermediate_dim, name='hidden_2',
                                   kernel_initializer=glorot_initializer,
                                   bias_initializer=normal_initializer))

        self.q_y_x_model.add(Activation('relu'))

        self.q_y_x_model.add(Dense(self.n_y, name='q_y_x_logit',
                                   kernel_initializer=normal_initializer,
                                   bias_initializer=normal_initializer))

        # Make q(z|x,y) network
        self.q_z_xy = Sequential()
        self.q_z_xy.add(Dense(self.intermediate_dim,
                              kernel_initializer=glorot_initializer,
                              bias_initializer=normal_initializer,
                              input_dim=self.n_x + self.n_y))

        self.q_z_xy.add(Activation('relu'))
        self.q_z_xy.add(Dense(self.intermediate_dim,
                              kernel_initializer=glorot_initializer,
                              bias_initializer=normal_initializer))

        self.q_z_xy.add(Activation('relu'))

        # which results in two networks, one for mean and one for log variance
        self.q_z_xy_mean = Sequential()
        self.q_z_xy_mean.add(self.q_z_xy)
        self.q_z_xy_mean.add(Dense(self.n_z,
                                   kernel_initializer=normal_initializer,
                                   bias_initializer=normal_initializer))

        self.q_z_xy_log_var = Sequential()
        self.q_z_xy_log_var.add(self.q_z_xy)
        self.q_z_xy_log_var.add(Dense(self.n_z,
                                      kernel_initializer=normal_initializer,
                                      bias_initializer=normal_initializer))
        # Make p(x|y,z) network - if Gaussain: two, one for mean one for log var
        #                     - if Bernoulii: just one for prob for each entry

        if self.x_dist == 'Gaussian':
            self.p_x_yz_model = Sequential()
            self.p_x_yz_model.add(Dense(self.intermediate_dim,
                                        kernel_initializer=glorot_initializer,
                                        bias_initializer=normal_initializer,
                                        input_dim=self.n_z + self.n_y))
            self.p_x_yz_model.add(Activation('relu'))
            self.p_x_yz_model.add(Dense(self.intermediate_dim,
                                        kernel_initializer=glorot_initializer,
                                        bias_initializer=normal_initializer))
            self.p_x_yz_model.add(Activation('relu'))

            self.p_x_yz_mean = Sequential()
            self.p_x_yz_mean.add(self.p_x_yz_model)
            self.p_x_yz_mean.add(Dense(self.n_x,
                                       kernel_initializer=normal_initializer,
                                       bias_initializer=normal_initializer))
            self.p_x_yz_log_var = Sequential()
            self.p_x_yz_log_var.add(self.p_x_yz_model)
            self.p_x_yz_log_var.add(Dense(self.n_x,
                                          kernel_initializer=normal_initializer,
                                          bias_initializer=normal_initializer))
            #
        elif self.x_dist == 'Bernoulli':
            self.p_x_z_mean=Sequential()
            self.p_x_z_mean.add(Dense(self.intermediate_dim,
                                      kernel_initializer=glorot_initializer,
                                      bias_initializer=normal_initializer,
                                      input_dim=self.n_z + self.n_y))
            self.p_x_z_mean.add(Activation('relu'))
            self.p_x_z_mean.add(Dense(self.intermediate_dim,
                                      kernel_initializer=glorot_initializer,
                                      bias_initializer=normal_initializer))
            self.p_x_z_mean.add(Activation('relu'))
            self.p_x_z_mean.add(Dense(self.n_x,
                                      kernel_initializer=normal_initializer,
                                      bias_initializer=normal_initializer))

    def compute_loss(self):
        """ manipulate computed components and compute loss """
        self.elbo_l = tf.reduce_mean(self.labelled_loss(self.x_l, self.y_l))
        self.qy_ll = tf.reduce_mean(self.qy_loss(self.x_l, self.y_l))
        self.elbo_u = tf.reduce_mean(self.unlabelled_loss(self.x_u))
        weight_priors = self.l2_reg * self.weight_prior() / self.reg_term
        return -(self.loss_ratio * self.elbo_l + self.elbo_u +
                 self.alpha * self.qy_ll + weight_priors)

    def compute_unsupervised_loss(self):
        """ manipulate computed components and compute unsup loss """
        self.elbo_u = tf.reduce_mean(self.unlabelled_loss(self.x_u))
        weight_priors = self.l2_reg * self.weight_prior() / self.reg_term
        return -(self.elbo_u + weight_priors)

    def compute_supervised_loss(self):
        """ manipulate computed components and compute loss """
        self.elbo_l = tf.reduce_mean(self.labelled_loss(self.x_l, self.y_l))
        self.qy_ll = tf.reduce_mean(self.qy_loss(self.x_l, self.y_l))
        weight_priors = self.l2_reg * self.weight_prior() / self.reg_term
        return -(self.elbo_l + self.alpha * self.qy_ll + weight_priors)

    def labelled_loss(self, x, y):
        z_m, z_lv, z = self.sample_z(x, y)
        x_ = tf.tile(tf.expand_dims(x, 0), [self.mc_samples, 1, 1])
        y_ = tf.tile(tf.expand_dims(y, 0), [self.mc_samples, 1, 1])
        return self.lowerBound(x_, y_, z, z_m, z_lv)

    def unlabelled_loss(self, x):
        qy_l = self.predict(x)
        x_r = tf.tile(x, [self.n_y, 1])
        y_u = tf.reshape(tf.tile(tf.eye(self.n_y), [1, tf.shape(x)[0]]),
                         [-1, self.n_y])
        n_u = tf.shape(x)[0]
        lb_u = tf.transpose(tf.reshape(self.labelled_loss(x_r, y_u),
                                       [self.n_y, n_u]))
        lb_u = tf.reduce_sum(qy_l * lb_u, axis=-1)
        qy_entropy = -tf.reduce_sum(qy_l * tf.log(qy_l + 1e-10), axis=-1)
        return lb_u + qy_entropy

    def lowerBound(self, x, y, z, z_m, z_lv):
        """ Compute densities and lower bound given all inputs 
        of shape: (mc_samps X n_obs X n_dim)
        """
        l_px = self.compute_logpx(x, y, z)
        l_py = dgm.multinoulliLogDensity(y, self.prior, on_priors=True)
        l_pz = dgm.standardNormalLogDensity(z)
        l_qz = dgm.gaussianLogDensity(z, z_m, z_lv)
        return tf.reduce_mean(l_px + l_py + l_pz - l_qz, axis=0)

    def qy_loss(self, x, y=None):
        y_ = self.q_y_x_model(x)
        if y is None:
                return dgm.multinoulliUniformLogDensity(y_)
        else:
                return dgm.multinoulliLogDensity(y, y_)

    def sample_z(self, x, y):
        l_qz_in = tf.concat([x, y], axis=-1)
        z_mean = dgm.forwardPass(self.q_z_xy_mean, l_qz_in)
        z_log_var = dgm.forwardPass(self.q_z_xy_log_var, l_qz_in)
        return z_mean, z_log_var, dgm.sampleNormal(z_mean, z_log_var,
                                                   self.mc_samples)

    def compute_logpx(self, x, y, z):
        px_in = tf.reshape(tf.concat([y, z], axis=-1), [-1, self.n_y + self.n_z])
        if self.x_dist == 'Gaussian':
            mean, log_var = self.p_x_yz_mean(px_in), self.p_x_yz_log_var(px_in)
            mean = tf.reshape(mean, [self.mc_samples, -1, self.n_x])
            log_var = tf.reshape(log_var, [self.mc_samples, -1, self.n_x])
            return dgm.gaussianLogDensity(x, mean, log_var)
        elif self.x_dist == 'Bernoulli':
            logits = self.p_x_z_mean(px_in)
            logits = tf.reshape(logits, [self.mc_samples, -1, self.n_x])
            return dgm.bernoulliLogDensity(x, logits)

    def predict(self, x):
        """ predict y for given x with q(y|x) """
        return tf.nn.softmax(self.q_y_x_model(x))


    #def training_fd(self, x_l, y_l, x_u):
    #    return {self.x_l: x_l, self.y_l: y_l, self.x_u: x_u, self.x: x_l, self.y: y_l, self.reg_term:self.n_train}

    def _printing_feed_dict(self, Data, x_l, x_u, y, eval_samps, binarize):
        fd = super(m2, self)._printing_feed_dict(Data, x_l, x_u, y,
                                                 eval_samps, binarize)
        fd[self.reg_term] = self.n_train
        return fd

    def print_verbose1(self, epoch, fd, sess):
        total, elbo_l, elbo_u, qy_ll, weight_priors = \
            sess.run([self.compute_loss(), self.elbo_l, self.elbo_u,
                      self.qy_ll, weight_priors], fd)
        train_acc, test_acc = sess.run([self.train_acc, self.test_acc], fd)  
        print("Epoch: {}: Total: {:5.3f}, labelled: {:5.3f}, Unlabelled: {:5.3f}, q_y_ll: {:5.3f}, weight_priors: {:5.3f}, Training: {:5.3f}, Testing: {:5.3f}".format(epoch, total, elbo_l, elbo_u, qy_ll, weight_priors, train_acc, test_acc))  

    def print_verbose2(self, epoch, fd, sess):
        total, elbo_l, elbo_u = sess.run([self.compute_loss(), self.elbo_l, self.elbo_u] ,fd)
        train_acc, test_acc = sess.run([self.train_acc, self.test_acc], fd)     
        print("Epoch: {}: Total: {:5.3f}, labelled: {:5.3f}, Unlabelled: {:5.3f}, Training: {:5.3f}, Testing: {:5.3f}".format(epoch, total, elbo_l, elbo_u, train_acc, test_acc)) 

    def print_verbose3(self, epoch):
           print("Epoch: {}: Total: {:5.3f}, Unlabelled: {:5.3f}, KL_y: {:5.3f}, TrainingAc: {:5.3f}, TestingAc: {:5.3f}, TrainingK: {:5.3f}, TestingK: {:5.3f}".format(epoch, sum(self.curve_array[epoch][1:3]), self.curve_array[epoch][2], self.curve_array[epoch][3], self.curve_array[epoch][0], self.curve_array[epoch][6], self.curve_array[epoch][12], self.curve_array[epoch][13])) 
