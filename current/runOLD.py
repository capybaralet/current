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
from pylearn2.datasets import cifar10
from optparse import OptionParser
from pylearn2.datasets import preprocessing

import pylearn2.utils.serial as serial
import os
import numpy as np



MAX_EPOCHS_UNSUPERVISED = 3



num_channels = 3
input_width = 32
# 2D inputs only, code may break in spots for other input_dim values
input_dim = 2

patch_width = 4
# assume 2D for next line
patch_shape = (patch_width, patch_width)
patches_per_image = 3

num_components = num_channels * input_width**input_dim
#num_components = 
# keep_var_fraction currently doesn't work properly because of a bug in pca.py
keep_var_fraction = .9

train_size = 4900

nhid = 40


# Not sure why this would need to go here... 
curruptor = BinomialCorruptor(corruption_level=0.5)


# Loads the preprocessed dataset, or does the preprocessing and saves the dataset.
# Currently using CIFAR10

def get_processed_dataset():

    train_path = 'pp_cifar10_train.pkl'
    test_path = 'pp_cifar10_test.pkl'

    if os.path.exists(train_path) and \
            os.path.exists(test_path):
        print 'loading preprocessed data'
        trainset = serial.load(train_path)
        testset = serial.load(test_path)

    else:
        print 'loading raw data...'
        trainset = cifar10.CIFAR10(which_set="train")
        testset =  cifar10.CIFAR10(which_set="test")
	
        # Here's our pipeline
	pipeline = preprocessing.Pipeline()
        pipeline.items.append(preprocessing.ExtractPatchesWithPosition(patch_shape=patch_shape, patches_per_image=patches_per_image))
        pipeline.items.append(preprocessing.GlobalContrastNormalization(sqrt_bias=10., use_std=True))
        pipeline.items.append(preprocessing.PCA(num_components = num_components, keep_var_fraction = keep_var_fraction))
        pipeline.items.append(preprocessing.ExtractPatchPairs(patches_per_image = patches_per_image, num_images = train_size, input_width = input_width))

        trainset.apply_preprocessor(preprocessor=pipeline, can_fit=True)

        serial.save('pp_cifar10_train.pkl', trainset)
        serial.save('pp_cifar10_test.pkl', testset)

        # this path will be used for visualizing weights after training is done
        trainset.yaml_src = '!pkl: "%s"' % train_path
        testset.yaml_src = '!pkl: "%s"' % test_path

    return trainset, testset




def main():


    # the data isn't going to be fully processed, so we may have to
    # do some stuff to the testset still to make it work.
    trainset, testset = get_processed_dataset()

    # Creating the patch-pairs:
    design_matrix = trainset.get_design_matrix()
    processed_patch_size = design_matrix.shape[1]

    num_images = train_size
    
    examples_per_image = patches_per_image * (patches_per_image-1)
    num_examples = examples_per_image * num_images

    stamps = trainset.stamps
    max_stamp = input_width - patch_width
    d_size = (2*max_stamp+1)**input_dim

    patch_pairs = np.zeros((num_examples, 2*processed_patch_size))
    distances = np.zeros((num_examples, input_dim))
    distances_onehot = np.zeros((num_examples, d_size))
    examples = np.zeros((num_examples, 2*processed_patch_size + d_size))

    nvis = 2*processed_patch_size + d_size


    def flatten_encoding(encoding, max_stamp):
        dims = len(encoding)
        flat_encoding = 0
        for i in xrange(dims-1):
             flat_encoding += encoding[i]
             flat_encoding *= max_stamp
        flat_encoding += encoding[-1]


    # Can be done without (or with less) for loops?
    print 'begin for loop'
    for i in xrange(num_images):
        if (i%1000 == 0):
            print i, '-th outer loop...'
        for j in xrange(patches_per_image):
            patch1_num = i * patches_per_image + j
            patch1_pos = stamps[patch1_num,:]
            for k in xrange(patches_per_image):
                example_num = i*examples_per_image + j*(patches_per_image-1) + k
                if (k > j):
                    example_num -= 1
                if (k != j):                    
                    patch2_num = i * patches_per_image + k
                    patch2_pos = stamps[patch2_num,:]
                    distance = patch1_pos - patch2_pos
                    distances[example_num] = distance
                    distance_encoding = distance + max_stamp
                    distance_encoding = flatten_encoding(distance_encoding, max_stamp)
                    distances_onehot[example_num, distance_encoding] = 1
                    p1 = design_matrix[patch1_num]
                    p2 = design_matrix[patch2_num]
                    patch_pairs[example_num] = np.hstack((p1, p2))
                    examples[example_num] = np.hstack((patch_pairs[example_num], distances_onehot[example_num]))
    print 'end for loop'

    trainset.set_design_matrix(examples)

    model = DenoisingAutoencoder(
                                 corruptor = BinomialCorruptor(corruption_level=0.5),
                                 nhid = nhid,
                                 nvis = nvis,
                                 act_enc = 'sigmoid',
                                 act_dec = 'sigmoid',
                                 irange = .01
                                 )

    algorithm = SGD(
                    learning_rate = 0.1,
                    cost =  MeanSquaredReconstructionError(),
                    batch_size =  100,
                    monitoring_batches = 10,
                    monitoring_dataset =  trainset,
                    termination_criterion = EpochCounter(max_epochs=MAX_EPOCHS_UNSUPERVISED),
                    update_callbacks =  None
                    )

    extensions = None

    trainer = Train(model = model,
                algorithm = algorithm,
                save_path='run.pkl',
                save_freq=1,
                extensions = extensions,
                dataset = trainset)

    trainer.main_loop()


if __name__ == '__main__':
    main()
