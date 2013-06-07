"""
Script to train denoising autoencoder on patch-pair, displacement vector data from
CIFAR10.  Uses SGD.

currently, going with one-hot encoding on signed pixel distances for displacement vectors
"""

from pylearn2.corruption import BinomialCorruptor
from pylearn2.corruption import GaussianCorruptor
from pylearn2.costs.mlp import Default
from pylearn2.models.autoencoder import Autoencoder, DenoisingAutoencoder
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.costs.autoencoder import MeanSquaredReconstructionError
from pylearn2.termination_criteria import EpochCounter
from pylearn2.datasets import cifar10
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.transformer_dataset import TransformerDataset
from pylearn2.training_algorithms.sgd import MonitorBasedLRAdjuster
from pylearn2.train import Train
from pylearn2.datasets import preprocessing

from optparse import OptionParser

import pylearn2.config.yaml_parse
import pylearn2.utils.serial as serial

import os
import copy

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm



MAX_EPOCHS_UNSUPERVISED = 4

num_channels = 3
input_width = 32
# 2D inputs only, code may break in spots for other input_dim values
input_dim = 2

patch_width = 6
# assume 2D for next line
patch_shape = (patch_width, patch_width)
patches_per_image = 3

num_components = num_channels * input_width**input_dim
#num_components = 19
keep_var_fraction = 1e4

# This is easy enough to change if we look at cifar10.CIFAR10
train_size = 10000
start = 0
stop = train_size

nhid = 2000

num2plot = 4


learning_rate = .02


# remember to modify this if you change parameters.
run_params = [MAX_EPOCHS_UNSUPERVISED, num_channels, input_width, input_dim, patch_width, patch_shape, patches_per_image, num_components, keep_var_fraction, train_size, start, stop, nhid, num2plot, learning_rate]
saved_run_params = []
if os.path.exists('saved_run_params.pkl'):
    saved_run_params = serial.load('saved_run_params.pkl')
new_params = False 
if run_params != saved_run_params:
    serial.save('saved_run_params.pkl', run_params)
    new_params = True


# Loads the preprocessed dataset, or does the preprocessing and saves the dataset.
# Only the trainset is processed by this function.
# Currently using CIFAR10.
#
# Note to self: ADD AN OPTION to not used preprocessed data, 
# also, by default, we save off parameters, and only load saved data if parameters are identical
# but it is buggy if the script doesn't finish running


def get_processed_dataset():

    train_path = 'pp_cifar10_train.pkl'
    test_path = 'pp_cifar10_test.pkl'


    if os.path.exists(train_path) and os.path.exists(test_path) and not new_params:
        print 'loading preprocessed data'
        trainset = serial.load(train_path)
        testset = serial.load(test_path)

    else:
        print 'loading raw data...'

        pipeline = preprocessing.Pipeline()
        pipeline.items.append(preprocessing.ExtractPatchesWithPosition(patch_shape=patch_shape, patches_per_image=patches_per_image))
        pipeline.items.append(preprocessing.GlobalContrastNormalization(sqrt_bias=10., use_std=True))
        pipeline.items.append(preprocessing.PCA(num_components = num_components, keep_var_fraction = keep_var_fraction))
        pipeline.items.append(preprocessing.ExtractPatchPairs(patches_per_image = patches_per_image, num_images = train_size, input_width = input_width))

        trainset = cifar10.CIFAR10(which_set="train", start = start, stop = stop)
        testset =  cifar10.CIFAR10(which_set="test")

        trainset.preprocessor = pipeline

        trainset.apply_preprocessor(preprocessor=pipeline, can_fit=True)


        # the pkl-ing is having issues, the dataset is maybe too big.
        serial.save(train_path, trainset)
        serial.save(test_path, testset)

        # this path will be used for visualizing weights after training is done
        trainset.yaml_src = '!pkl: "%s"' % train_path
        testset.yaml_src = '!pkl: "%s"' % test_path

    return trainset, testset






#def reconstruct_d(v,...):
    # Returns a reconstructed displacement vector from a corresponding output encoding.
    #
    # Currently, our d-vectors are onehot as inputs. We look for the maximum index of 
    # the output encoding and chose the corresponding d for out reconstructed displacement.
    
    
def add_outlines(images_data, x, y, pw, color = 'white'):
    if color == 'black':
        b = 0.0
    else:
        b = 1.0
    image_width = images_data.shape[1]
    x_end = x+pw
    y_end = y+pw
    lx = len(x)
    ly = len(y)
    assert lx == ly
    for i in range(lx):
        xi = x[i].astype('int')
        yi = y[i].astype('int')
        xi_end = min(x_end[i].astype('int'), image_width-1)
        yi_end = min(y_end[i].astype('int'), image_width-1)
        xr = range(xi-1, xi_end+1)
        yr = range(yi-1, yi_end+1)
        images_data[i, xi_end, yr,:] = b
        images_data[i, xr, yi_end,:] = b
        if xi != 0:
             images_data[i, xi-1, yr,:] = b
        if yi != 0:
             images_data[i, xr, yi-1,:] = b        


def flat_to_2D(n, d_shape):
    input_dim = len(d_shape)
    enc = np.zeros((len(n), input_dim))
    temp = n
    for i in range(input_dim):
        ds = d_shape[-i-1]
        enc[:,i] = (temp % ds) - 1 - ds/2
        temp -= enc[:,i]
        temp /= ds
    return enc


def insertpatches(outs, patches, x, y, pw):
    x_end = x+pw
    y_end = y+pw
    lx = len(x)
    ly = len(y)
    assert lx == ly
    for i in range(lx):
        xi = x[i].astype('int')
        yi = y[i].astype('int')
        xi_end = x_end[i].astype('int')
        yi_end = y_end[i].astype('int')
        xr = range(xi, xi_end)
        yr = range(yi, yi_end)
        outs[i, xi:xi_end, yi:yi_end] = patches[i]


def pp_to_patches_num(pp, patches_per_image = patches_per_image):
    ppi = patches_per_image
    pp_per_image = ppi * (ppi - 1)
    image_num = pp / pp_per_image
    pp_num = pp % pp_per_image
    p1_num = pp_num / (ppi - 1)
    p2_num = pp_num % (ppi - 1)
    if p2_num >= p1_num:
        p2_num += 1
    p1_num += pp_per_image*image_num
    p2_num += pp_per_image*image_num
    return p1_num, p2_num
    

# Only works on lists indexed by the entire set of patches
def expand_p1(plist, patches_per_image = patches_per_image):
    return np.repeat(plist, patches_per_image-1)

# Only works on lists indexed by the entire set of patches
def expand_p2(plist, patches_per_image = patches_per_image, num_images = train_size):
    ppi = patches_per_image
    pl = []
    for i in range(num_images):
        pli = plist[ppi*i: ppi*(i+1) - 1]
        pli = np.tile(pli, ppi)
        pl = np.hstack((pl,pli))
    l = []
    for i in range(ppi):
        l = np.hstack((l, np.zeros(i)))
        l = np.hstack((l, np.ones(ppi - i - 1)))
    ll = np.tile(l, num_images)
    rval = pl+ll
    return rval
        

def main():

    # Only the trainset is processed by this function.
    print 'getting preprocessed data for training model'
    pp_trainset, testset = get_processed_dataset()
    # remember to change here when changing datasets 
    print 'loading unprocessed data for input displays'
    trainset = cifar10.CIFAR10(which_set="train")
    
    dmat = pp_trainset.get_design_matrix()
    nvis = dmat.shape[1]

    model = DenoisingAutoencoder(
                                 corruptor = BinomialCorruptor(corruption_level=0.3),
                                 nhid = nhid,
                                 nvis = nvis,
                                 act_enc = 'sigmoid',
                                 act_dec = 'sigmoid',
                                 irange = .01
                                 )

    algorithm = SGD(
                    learning_rate = learning_rate,
                    cost =  MeanSquaredReconstructionError(),
                    batch_size =  100,
                    monitoring_batches = 10,
                    monitoring_dataset =  pp_trainset,
                    termination_criterion = EpochCounter(max_epochs=MAX_EPOCHS_UNSUPERVISED),
                    update_callbacks =  None
                    )

    extensions = None

    trainer = Train(model = model,
                algorithm = algorithm,
                save_path='run.pkl',
                save_freq=1,
                extensions = extensions,
                dataset = pp_trainset)

    trainer.main_loop()

    ####################
    # Plot and Save:

    # choose random patch-pairs to plot
    stamps = pp_trainset.stamps
    num_examples = stamps.shape[0]
    to_plot = np.random.randint(0, num_examples, num2plot)
    
    # use to_plot indices to extract data
    stamps_data = stamps[to_plot]
    image_numbers = stamps[to_plot,0].astype(int)
    X = trainset.X
    images_data = trainset.get_topological_view(X[image_numbers])
    p1x = stamps_data[:,1]
    p1y = stamps_data[:,2]
    p2x = stamps_data[:,3]
    p2y = stamps_data[:,4]

    # For input ppd's, once we've identified the patches, we just outline them and draw an arrow for d 
    # This might mess with original trainset (I dunno), in which case, we should make a copy 
    add_outlines(images_data, p1x, p1y, patch_width)
    add_outlines(images_data, p2x, p2y, patch_width)

    ##################################################
    # translating outputs back into things we can plot
    dataset = pp_trainset
    Xout = dataset.X.astype('float32')
    max_stamp = input_width - patch_width
    d_size = (2*max_stamp+1)**input_dim
    # displacement
    d_enc = Xout[:,-d_size:]
    d_out_flat = np.argmax(d_enc, axis = 1)
    d_shape = [2*max_stamp+1, 2*max_stamp+1] # assumed 2D
    d_out = flat_to_2D(d_out_flat, d_shape)
    d_out[to_plot, ]
    # patches
    vc = dataset.view_converter
    p_enc = Xout[:,:len(Xout.T)-d_size]
    p_size = p_enc.shape[1]/2
    p1_enc = p_enc[:,:p_size]
    p2_enc = p_enc[:,p_size:]
    p1_enc = vc.design_mat_to_topo_view(p1_enc)
    p2_enc = vc.design_mat_to_topo_view(p2_enc)
    pp = dataset.preprocessor
    gcn = pp.items[1]
    means = gcn.means
    normalizers = gcn.normalizers
    toshape = (num_examples,)
    for i in range(input_dim):
        toshape += (1,)
    if num_channels != 1:
        toshape += (1,)
    # When the number of patches and patch-pairs differs, this breaks.  
    # I need to match up normalizers/means with their corresponding patches
    # undoing the PCA might be breaking too, but without errors...
    normalizers1 = expand_p1(normalizers)
    normalizers2 = expand_p2(normalizers)    
    means1 = expand_p1(means)
    means2 = expand_p2(means)

    p1_enc *= normalizers1.reshape(toshape)
    p1_enc += means1.reshape(toshape)
    p2_enc *= normalizers2.reshape(toshape)
    p2_enc += means2.reshape(toshape)
    # Now, we pull off the same examples from the data to compare to dAE inputs in plots
    outputs = copy.deepcopy(images_data)
    insertpatches(outputs, p1_enc[to_plot], p1x, p1y, patch_width)
    insertpatches(outputs, p2_enc[to_plot], p2x, p2y, patch_width)
    

    plt.figure()

    for i in range(num2plot):
        # Inputs
        plt.subplot(num2plot,2,2*i+1)
        plt.imshow(images_data[i], cmap = cm.Greys_r)
        print stamps_data[i]
        a = (stamps_data[i,2]+patch_width/2, stamps_data[i,1]+patch_width/2, 
             stamps_data[i,6], stamps_data[i,5]
            )
        plt.arrow(a[0],a[1],a[2],a[3], head_width = 1.0, head_length = 0.6)
        # Outputs
        plt.subplot(num2plot,2,2*i+2)
        plt.imshow(outputs[i], cmap = cm.Greys_r)
        plt.arrow(a[0], a[1], d_out[to_plot[i],1], d_out[to_plot[i],0], head_width = 1.0, head_length = 0.6)

    plt.show()


    savestr = 'cifar_ppd.png'
    plt.savefig(savestr)

if __name__ == '__main__':
    main()
