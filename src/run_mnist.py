import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import rc
import pickle, sys, pdb, gzip
if sys.version_info[0]<3:
    import cPickle

import numpy as np
from sklearn.metrics import log_loss, confusion_matrix
from sklearn.manifold import TSNE as tsne
import tensorflow as tf
from data.SSL_DATA import SSL_DATA
from data.mnist import mnist
from data.data import create_semisupervised, encode_onehot, load_dataset, make_dataset
from models.m2 import m2
from models.m2_k import m2_k
from models.gm_dgm import gm_dgm
from models.adgm_k import adgm_k
from models.agm_dgm import agm_dgm
from keras.datasets import cifar10, cifar100
import argparse
import os

# from models.adgm import adgm
# from models.sdgm import sdgm
# from models.blendedv2 import blended
# from models.sblended import sblended
# from models.b_blended import b_blended

### Script to run an MNIST experiment with generative SSL models ###

## argv[1] - proportion of training data labeled (or for mnist, number of labels from each class)
## argv[2] - Dataset seed
## argv[3] - noise level in moons dataset / Threshold for reduction in mnist
## argv[4] - model to use (m2, adgm, sdgm, sslpe, b_sdgm, msdgm)
## argv[5] - number of runs per beta
## argv[6] - number of MC samples

# Experiment parameters
#num_labeled, threshold = int(sys.argv[1]), float(sys.argv[3])

def get_args():
    '''This function parses and return arguments passed in'''
    # Assign description to the help doc
    parser = argparse.ArgumentParser(
        description='Run different models over different datasets with different props labelled')
    # Add arguments
    parser.add_argument(
        '-m', '--model_name', choices=['m2', 'm2_k', 'gm_dgm','gmm_vae', 'adgm_k', 'agm_dgm'], required=True)
    parser.add_argument(
        '-d', '--dataset', choices=['mnist', 'svhn', 'cifar10', 'cifar100', 'activity'], required=True)
    parser.add_argument(
        '-p', '--prop_labelled', type=float, required=True)
    parser.add_argument(
        '-r', '--number_of_runs', type=int, required=True)
    parser.add_argument(
        '-e', '--number_of_epochs', type=int, required=True)
    # Array for all arguments passed to script
    args = parser.parse_args()
    # Assign args to variables
    return args

args=get_args()

model_name = args.model_name
#model_name = 'm2'
dataset_name = args.dataset
#dataset_name = 'svhn'
prop_labelled = args.prop_labelled
#prop_labelled=0.1
num_runs = args.number_of_runs
#num_runs=10
n_epochs = args.number_of_epochs
#n_epochs=10
if prop_labelled == 0:
    learning_paradigm = 'unsupervised'
elif prop_labelled < 0:
    learning_paradigm = 'supervised'
elif prop_labelled > 0:
    learning_paradigm = 'semisupervised'
# Load and conver data to relevant type

token_list = map(str, args.__dict__.values())
token_list.reverse()
token = "-".join(token_list)
#token='test'

#make output directory if doesnt exist
output_dir =  os.path.join('/jmain01/home/JAD017/sjr01/mxw35-sjr01/Projects/CVAE/output/results', model_name ,dataset_name)

if os.path.isdir(output_dir) == False:
    os.makedirs(output_dir)


x_train, y_train, x_test, y_test, binarize, x_dist, n_y, n_x, f_enc, f_dec  = load_dataset(dataset_name, preproc=True)

num_labelled = int(prop_labelled*x_train.shape[0])

Data = make_dataset(learning_paradigm, x_train, y_train, x_test, y_test, dataset_name, num_labelled=num_labelled)
if dataset_name=='activity':
    Data = make_dataset(learning_paradigm, x_train=[], y_train=[], x_test=x_test, y_test=y_test, dataset_name=dataset_name, num_labelled=num_labelled, do_split = False, x_labelled=x_train[0], y_labelled=y_train[0], x_unlabelled=x_train[1], y_unlabelled=y_train[1])
else:
    Data = make_dataset(learning_paradigm, x_train, y_train, x_test, y_test, dataset_name, num_labelled=num_labelled)

if prop_labelled<0.002 and prop_labelled>0:
    l_bs, u_bs = 10,100
    alpha = 0.02
else:
    l_bs, u_bs = 100,100
    alpha = 0.1

loss_ratio = float(l_bs)/float(u_bs)

# Specify model parameters
lr = (3e-4,)
n_z, n_a = 100, 100
n_w = 50
n_hidden = [500, 500]
temp_epochs, start_temp = None, 0.0
l2_reg, initVar, alpha = .5, -10., 1.1
#batchnorm, mc_samps = True, int(sys.argv[6])
batchnorm, mc_samps = False, 1

eval_samps = 1000

logging, verbose = False, 2

np.random.seed(seed=0)

Data.reset_counters()
results=[]
for i in range(num_runs):
    print("Starting work on run: {}".format(i))
    Data.reset_counters()
    np.random.seed(2)
    tf.set_random_seed(2)
    tf.reset_default_graph()
    model_token = token+'-'+str(i)
    if model_name == 'm2':
        model = m2(n_x, n_y, n_z, n_hidden, x_dist=x_dist, batchnorm=batchnorm, mc_samples=mc_samps, l2_reg=l2_reg, learning_paradigm=learning_paradigm, name=model_token, ckpt = model_token)
    #
    if model_name == 'm2_k':
        model = m2_k(n_x, n_y, n_z, n_hidden, x_dist=x_dist, batchnorm=batchnorm, mc_samples=mc_samps, l2_reg=l2_reg, learning_paradigm=learning_paradigm, name=model_token, ckpt = model_token, loss_ratio=loss_ratio)
    #
    if model_name == 'gm_dgm':
        model = gm_dgm(n_x, n_y, n_z, n_hidden, x_dist=x_dist, batchnorm=batchnorm, mc_samples=mc_samps, l2_reg=l2_reg, learning_paradigm=learning_paradigm, name=model_token, ckpt = model_token, loss_ratio=loss_ratio)
    #
    if model_name == 'gmm_vae':
        model = gmm_vae(n_x, n_y, n_w, n_z, n_hidden, x_dist=x_dist, batchnorm=batchnorm, mc_samples=mc_samps, l2_reg=l2_reg, learning_paradigm=learning_paradigm, name=model_token, ckpt = model_token)
    #
    if model_name == 'dgmm_z':
        model = dgmm_z(n_x, n_y, n_z1, n_z2, n_hidden, x_dist=x_dist, batchnorm=batchnorm, mc_samples=mc_samps, l2_reg=l2_reg, learning_paradigm=learning_paradigm, name=model_token, ckpt = model_token)
    #
    if model_name == 'adgm_k':
        model = adgm_k(n_x, n_y, n_z, n_a, n_hidden, x_dist=x_dist, batchnorm=batchnorm, mc_samples=mc_samps, l2_reg=l2_reg, learning_paradigm=learning_paradigm, name=model_token, ckpt = model_token)
    #
    if model_name == 'agm_dgm':
        model = agm_dgm(n_x, n_y, n_z, n_a, n_hidden, x_dist=x_dist, batchnorm=batchnorm, mc_samples=mc_samps, l2_reg=l2_reg, learning_paradigm=learning_paradigm, name=model_token, ckpt = model_token)
    #
    model.train(Data, n_epochs, l_bs, u_bs, lr, eval_samps=eval_samps, temp_epochs=temp_epochs, start_temp=start_temp, binarize=binarize, logging=logging, verbose=verbose)
    results.append(model.curve_array)
    np.save(os.path.join(output_dir,'curve_'+token+'_'+str(i)+'.npy'), model.curve_array)

    if learning_paradigm == 'semisupervised':
        Data.recreate_semisupervised(i)

np.save(os.path.join(output_dir,'results_'+ token+'.npy'), results)
