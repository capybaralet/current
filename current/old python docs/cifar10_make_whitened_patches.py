"""
This script contains a pipeline consisting of:
1. Global Contrast Normalization
2. PCA whitening, with a variance-based cutoff 
3. (patch-pair, displacement vector) extraction

Currently, it operates on CIFAR10 data.
"""

# Code Base: https://github.com/lisa-lab/pylearn2/blob/0bc45116a2c85b9ae157811bf69dfd929363f2a6/pylearn2/scripts/lcc_tangents/make_dataset.py

# INSERT imports

# Parameters for patchsize, etc.
patch_size = 8
stride = 1
panels = 3
num_channels = 3
num_components = num_channels*patch_size**2
keep_var_fraction = .99


train = cifar10.CIFAR10(which_set="train",center=True)
test = cifar10.CIFAR10(which_set="test")

pipeline = preprocessing.Pipeline()
# Uses Coates, Lee, and Ng's contrast normalization settings
pipeline.items.append(preprocessing.GlobalContrastNormalization(sqrt_bias=10.0, use_std=True))
pipeline.items.append(preprocessing.PCA(num_components = num_components, keep_var_fraction))
# EDIT HERE (ExtractGridPatchPairs needs to be implemented in preprocessing.py)
pipeline.items.append(preprocessing.ExtractGridPatchPairs( (patch_size, patch_size), (stride, stride), (panels, panels) ))




train.apply_preprocessor(preprocessor = pipeline, can_fit = True)
test.apply_preprocessor(preprocessor = pipeline, can_fit = False)

serial.save('cifar10_preprocessed_train.pkl',train)
serial.save('cifar10_preprocessed_test.pkl',test)






###########################################################
# BELOW HERE IS OLD


import numpy as np

from pylearn2.utils import serial
from pylearn2.utils import string_utils
from pylearn2.datasets import preprocessing
from pylearn2.datasets import cifar10
import pylearn2.pca as pca

output_dir = string_utils.preprocess('/u/kruegers/repo/current/pylearn2/pylearn2/datasets/cifar10')

print "Preparing output directory..."
serial.mkdir( output_dir )


print 'Loading CIFAR-10 train and test datasets...'
trainset = cifar10.CIFAR10(which_set = 'train')
testset = cifar10.CIFAR10(which_set = 'test')

print "Learning the preprocessor"
preprocessor = pca.PCA()

print "Preprocessing the unsupervised train data..."
trainset.apply_preprocessor(preprocessor = preprocessor, can_fit = True)
print 'Saving the unsupervised train data'
trainset.use_design_loc(output_dir+'/train.npy')
serial.save(output_dir + '/train.pkl', trainset)

print "Preprocessing the test data..."
testset.apply_preprocessor(preprocessor = preprocessor, can_fit = False)
print "Saving the test data"
testset.use_design_loc(output_dir+'/test.npy')
serial.save(output_dir+'/test.pkl', testset)

print "Saving the preprocessor"
serial.save(output_dir + '/preprocessor.pkl',preprocessor)

