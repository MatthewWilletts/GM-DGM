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
from data.data import create_semisupervised, encode_onehot, load_dataset, make_dataset, get_non_wear_indices, load_feat_csv_file
from models.m2 import m2
from models.m2_k import m2_k
from models.gm_dgm import gm_dgm
from models.gm_dgm_conv import gm_dgm_conv
from models.adgm_k import adgm_k
from models.agm_dgm import agm_dgm
from keras.datasets import cifar10, cifar100
from utils.checkmate import get_best_checkpoint
from sklearn.metrics import cohen_kappa_score

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

def predict_new(x):
    saver = tf.train.Saver()
    with tf.Session() as session:
        ckpt = get_best_checkpoint(model.ckpt_dir)
        saver.restore(session, ckpt)
        if model_name == 'm2':
            pred = session.run([model.predictions], {model.x:x})
        else:
            y_ = model.q_y_x_model(model.x)
            pred = session.run([y_], {model.x:x})
    return pred


def get_args():
    '''This function parses and return arguments passed in'''
    # Assign description to the help doc
    parser = argparse.ArgumentParser(
        description='Run different models over different datasets with different props labelled')
    # Add arguments
    parser.add_argument(
        '-m', '--model_name', choices=['m2', 'm2_k', 'gm_dgm','gmm_vae', 'adgm_k', 'agm_dgm', 'gm_dgm_conv'], required=True)
    parser.add_argument(
        '-d', '--dataset', choices=['mnist', 'svhn', 'cifar10', 'cifar100', 'activity', 'biobank', 'activity_basic'], required=True)
    parser.add_argument(
        '-p', '--prop_labelled', type=float, required=True)
    parser.add_argument(
        '-r', '--number_of_runs', type=int, required=True)
    parser.add_argument(
        '-e', '--number_of_epochs', type=int, required=True)
    parser.add_argument(
        '-c', '--classes_to_hide', nargs='*', type=int)
    parser.add_argument(
        '-a', '--number_of_classes_to_add', type=int, default=0)
    parser.add_argument(
        '-z', '--number_of_dims_z', type=int, default=100)
    # Array for all arguments passed to script
    args = parser.parse_args()
    # Assign args to variables
    return args

args=get_args()

print(args)

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
classes_to_hide = args.classes_to_hide
#classes_to_hide = [7,8,9]
number_of_classes_to_add = args.number_of_classes_to_add
n_z = args.number_of_dims_z
#number_of_classes_to_add = 3
if dataset_name == 'activity':
    learning_paradigm = 'un-semisupervised'
elif prop_labelled == 0:
    learning_paradigm = 'unsupervised'
elif prop_labelled < 0:
    learning_paradigm = 'supervised'
elif prop_labelled > 0 and classes_to_hide is not None:
    learning_paradigm = 'un-semisupervised'
elif prop_labelled > 0:
    learning_paradigm = 'semisupervised'
# Load and conver data to relevant type
print(learning_paradigm)


token_list = map(str, args.__dict__.values())
token_list.reverse()
token = "-".join(token_list)

token = token.replace('[','c')
token = token.replace(']','')
token = token.replace(' ','_')
token = token.replace(',','')

#make output directory if doesnt exist
output_dir =  os.path.join('/jmain01/home/JAD017/sjr01/mxw35-sjr01/Projects/CVAE/output/results_masked', model_name ,dataset_name)

if os.path.isdir(output_dir) == False:
    os.makedirs(output_dir)


x_train, y_train, x_test, y_test, binarize, x_dist, n_y, n_x, f_enc, f_dec  = load_dataset(dataset_name, preproc=True, bio_t=[i for i in range(40)])

if dataset_name == 'activity' or dataset_name == 'biobank' or dataset_name == 'activity_basic':
    num_labelled=None
    num_classes = y_train[0][0].shape[1]
else:
    num_labelled = int(prop_labelled*x_train.shape[0])
    num_classes = y_train.shape[1]

if dataset_name == 'activity' or dataset_name == 'biobank':
    classes_to_hide=[1]
elif dataset_name == 'activity_basic':
    classes_to_hide=[7]

#remove certain classes:
if classes_to_hide is not None and dataset_name not in ['activity', 'biobank','activity_basic']:
    num_labelled = [int(float(num_labelled)/num_classes)]*num_classes
    for hide_class in classes_to_hide:
        num_labelled[hide_class] = 0

prior = np.array([1.0/n_y]*n_y)

if classes_to_hide is not None and dataset_name not in ['activity', 'biobank','activity_basic']:
    prior_for_other_classes = (1.0 - (float(n_y)-float(len(classes_to_hide)))/float(n_y))/(float(len(classes_to_hide))+float(number_of_classes_to_add))
    for hide_class in classes_to_hide:
        prior[hide_class] = prior_for_other_classes
    if number_of_classes_to_add >0:
        prior = np.concatenate((prior, np.ones(number_of_classes_to_add)*prior_for_other_classes))
elif classes_to_hide is not None and dataset_name in ['activity', 'biobank','activity_basic']:
    weight_for_old_classes = 0.5
    empricial_counts = np.array(Counter(y_train[0][0].argmax(1)).values(), 'float')/sum(Counter(y_train[0][0].argmax(1)).values())
    prior_for_other_classes = (1-weight_for_old_classes)/float(number_of_classes_to_add+len(classes_to_hide))
    mask = np.ones(prior.shape,dtype=bool)
    mask[classes_to_hide] = 0
    prior[mask] = weight_for_old_classes * empricial_counts
    for hide_class in classes_to_hide:
        prior[hide_class] = prior_for_other_classes
    if number_of_classes_to_add >0:
        prior = np.concatenate((prior, np.ones(number_of_classes_to_add)*prior_for_other_classes))



n_y = n_y + number_of_classes_to_add

if dataset_name=='activity' or dataset_name=='biobank' or dataset_name == 'activity_basic':
    Data = make_dataset(learning_paradigm, x_train=[], y_train=[], x_test=x_test, y_test=y_test[0], dataset_name=dataset_name, num_labelled=num_labelled, number_of_classes_to_add=number_of_classes_to_add, do_split = False, x_labelled=x_train[0], y_labelled=y_train[0][0], x_unlabelled=x_train[1], y_unlabelled=y_train[0][1])
else:
    Data = make_dataset(learning_paradigm, x_train, y_train, x_test, y_test, dataset_name, num_labelled=num_labelled, number_of_classes_to_add=number_of_classes_to_add)


if prop_labelled<0.002 and prop_labelled>0:
    l_bs, u_bs = 10,100
    alpha = 0.02
else:
    l_bs, u_bs = 111,111
    alpha = 0.1

loss_ratio = float(l_bs)/float(u_bs)

# Specify model parameters
lr = (3e-4,)
n_a = 100
n_w = 50
n_hidden = [500, 500]
temp_epochs, start_temp = None, 0.0
l2_reg, initVar, alpha = .5, -10., 1.1
#batchnorm, mc_samps = True, int(sys.argv[6])
batchnorm, mc_samps = False, 1

eval_samps = 1000

logging, verbose = False, 3

np.random.seed(seed=0)

Data.reset_counters()
results=[]
for i in range(num_runs):
    print("Starting work on run: {}".format(i))
    Data.reset_counters()
    np.random.seed(2)
    tf.set_random_seed(2)
    tf.reset_default_graph()
    model_token = token+'-'+str(i)+'---'
    if model_name == 'm2':
        model = m2(n_x, n_y, n_z, n_hidden, x_dist=x_dist, batchnorm=batchnorm, mc_samples=mc_samps, l2_reg=l2_reg, learning_paradigm=learning_paradigm, name=model_token, ckpt = model_token)
    #
    if model_name == 'm2_k':
        model = m2_k(n_x, n_y, n_z, n_hidden, x_dist=x_dist, batchnorm=batchnorm, alpha=alpha, mc_samples=mc_samps, l2_reg=l2_reg, learning_paradigm=learning_paradigm, name=model_token, ckpt = model_token, prior=prior, loss_ratio=loss_ratio, output_dir=output_dir)
    #
    if model_name == 'gm_dgm':
        model = gm_dgm(n_x, n_y, n_z, n_hidden, x_dist=x_dist, batchnorm=batchnorm, alpha=alpha, mc_samples=mc_samps, l2_reg=l2_reg, learning_paradigm=learning_paradigm, name=model_token, ckpt = model_token, prior=prior[0:n_y]/float(sum(prior[0:n_y])), loss_ratio=loss_ratio, output_dir=output_dir)
    #
    if model_name == 'gm_dgm_conv':
        model = gm_dgm_conv(n_x, n_y, n_z, n_hidden, x_dist=x_dist, batchnorm=batchnorm, alpha=alpha, mc_samples=mc_samps, l2_reg=l2_reg, learning_paradigm=learning_paradigm, name=model_token, ckpt = model_token, prior=prior[0:n_y]/float(sum(prior[0:n_y])), loss_ratio=loss_ratio, output_dir=output_dir)
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
    model.loss = model.compute_loss()
    model.train(Data, n_epochs, l_bs, u_bs, lr, eval_samps=eval_samps, temp_epochs=temp_epochs, start_temp=start_temp, binarize=binarize, logging=logging, verbose=verbose)
    results.append(model.curve_array)
    np.save(os.path.join(output_dir,'curve_'+token+'_'+str(i)+'.npy'), model.curve_array)
    y_pred_test = predict_new(Data.data['x_test'])[0]
    conf_mat = confusion_matrix(Data.data['y_test'].argmax(1), y_pred_test.argmax(1))
    np.save(os.path.join(output_dir,'conf_mat_'+token+'_'+str(i)+'.npy'), conf_mat)
    np.savez(os.path.join(output_dir,'y_preds_labels_'+token+'_'+str(i)+'.npz'), y_true=Data.data['y_test'].argmax(1), y_pred=y_pred_test.argmax(1), y_labels = y_test[1])
    if learning_paradigm == 'semisupervised' or learning_paradigm == 'un-semisupervised':
        Data.recreate_semisupervised(i)

np.save(os.path.join(output_dir,'results_'+ token+'.npy'), results)
