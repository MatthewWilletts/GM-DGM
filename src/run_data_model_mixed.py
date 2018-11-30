import matplotlib
from collections import Counter
import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from data.data import load_dataset, make_dataset
from models.m2 import m2
from models.gm_dgm import gm_dgm
from utils.checkmate import get_best_checkpoint
import argparse
import os

matplotlib.use('Agg')
np.random.seed(seed=0)

# from models.adgm import adgm
# from models.sdgm import sdgm
# from models.blendedv2 import blended
# from models.sblended import sblended
# from models.b_blended import b_blended

### Script to run an MNIST experiment with generative SSL models ###


def predict_new(x):
    saver = tf.train.Saver()
    with tf.Session() as session:
        ckpt = get_best_checkpoint(model.ckpt_dir)
        saver.restore(session, ckpt)
        if model_name == 'm2':
            pred = session.run([model.predictions], {model.x: x})
        else:
            y_ = model.q_y_x_model(model.x)
            pred = session.run([y_], {model.x: x})
    return pred


def get_args():
    '''This function parses and return arguments passed in'''
    # Assign description to the help doc
    parser = argparse.ArgumentParser(
        description='Run DGM models over MNIST w. different prop labelled')
    # Add arguments
    parser.add_argument(
        '-m', '--model_name', choices=['m2', 'gm_dgm'], required=True)
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


args = get_args()

print(args)

model_name = args.model_name
dataset_name = 'mnist'
prop_labelled = args.prop_labelled
num_runs = args.number_of_runs
n_epochs = args.number_of_epochs
classes_to_hide = args.classes_to_hide
n_cth = float(len(classes_to_hide))
number_of_classes_to_add = args.number_of_classes_to_add
n_cta = float(number_of_classes_to_add)
n_z = args.number_of_dims_z

if prop_labelled == 0:
    learning_paradigm = 'unsupervised'
elif prop_labelled < 0:
    learning_paradigm = 'supervised'
elif prop_labelled > 0 and classes_to_hide is not None:
    learning_paradigm = 'semi-unsupervised'
elif prop_labelled > 0:
    learning_paradigm = 'semisupervised'
# Load and conver data to relevant type
print(learning_paradigm)


token_list = map(str, args.__dict__.values())
token_list.reverse()
token = "-".join(token_list)

token = token.replace('[', 'c')
token = token.replace(']', '')
token = token.replace(' ', '_')
token = token.replace(',', '')

#make output directory if doesnt exist
cwd = os.getcwd()
output_dir =  os.path.join(gi/output/results_masked', model_name ,dataset_name)

if os.path.isdir(output_dir) == False:
    os.makedirs(output_dir)


x_train, y_train, x_test, y_test, binarize, x_dist, n_y, n_x, f_enc, f_dec  = load_dataset(dataset_name, preproc=True, bio_t=[i for i in range(40)])


num_labelled = int(prop_labelled*x_train.shape[0])
num_classes = y_train.shape[1]

#remove certain classes:
if classes_to_hide is not None:
    num_labelled = [int(float(num_labelled)/num_classes)]*num_classes
    for hide_class in classes_to_hide:
        num_labelled[hide_class] = 0

prior = np.array([1.0 / n_y] * n_y)

n_y_f = float(n_y)

# Our option to add extra classes enables us to have the dimensionality of our
# y representation to be greater than the number of true classes - later on we
# will then associate all these unsupervised classes using a `cluster&label'
# approach.

# Thus we much add appropriate ammount of probability mass to these new classes
# so we divide the mass for the hidden (ie unsupervised) classes equally
# between the original, now unsupervised, classes and the extra, added, classes

if classes_to_hide is not None:
    prior_for_other_classes = (1.0 - (n_y_f-n_cth)/n_y_f)/(n_cth+n_cta)
    for hide_class in classes_to_hide:
        prior[hide_class] = prior_for_other_classes
    if number_of_classes_to_add >0:
        prior = np.concatenate((prior, np.ones(number_of_classes_to_add)*prior_for_other_classes))


n_y = n_y + number_of_classes_to_add


Data = make_dataset(learning_paradigm, x_train, y_train, x_test, y_test, dataset_name, num_labelled=num_labelled, number_of_classes_to_add=number_of_classes_to_add)



l_bs, u_bs = 100, 100
alpha = 0.1

loss_ratio = float(l_bs) / float(u_bs)

# Specify model parameters
lr = (3e-4,)
n_a = 100
n_w = 50
n_hidden = [500, 500]
l2_reg, alpha = .5, 1.1
mc_samps = 1
eval_samps = 1000
verbose = 3

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
    if model_name == 'gm_dgm':
        model = gm_dgm(n_x, n_y, n_z, n_hidden, x_dist=x_dist, batchnorm=batchnorm, alpha=alpha, mc_samples=mc_samps, l2_reg=l2_reg, learning_paradigm=learning_paradigm, name=model_token, ckpt = model_token, prior=prior[0:n_y]/float(sum(prior[0:n_y])), loss_ratio=loss_ratio, output_dir=output_dir)

    if learning_paradigm == 'semisupervised' or 'semi-unsupervised':
        model.loss = model.compute_loss()
    elif learning_paradigm == 'unsupervised':
        model.loss = model.compute_unsupervised_loss()
    elif model.learning_paradigm == 'supervised':
        model.loss = model.compute_supervised_loss()

    model.train(Data, n_epochs, l_bs, u_bs, lr, eval_samps=eval_samps, binarize=binarize, verbose=1)
    results.append(model.curve_array)
    np.save(os.path.join(output_dir,'curve_'+token+'_'+str(i)+'.npy'), model.curve_array)
    y_pred_test = predict_new(Data.data['x_test'])[0]
    conf_mat = confusion_matrix(Data.data['y_test'].argmax(1), y_pred_test.argmax(1))
    np.save(os.path.join(output_dir,'conf_mat_'+token+'_'+str(i)+'.npy'), conf_mat)
    np.savez(os.path.join(output_dir,'y_preds_labels_'+token+'_'+str(i)+'.npz'), y_true=Data.data['y_test'].argmax(1), y_pred=y_pred_test.argmax(1), y_labels = y_test[1])
    if learning_paradigm == 'semisupervised' or learning_paradigm == 'un-semisupervised':
        Data.recreate_semisupervised(i)

np.save(os.path.join(output_dir,'results_'+ token+'.npy'), results)
