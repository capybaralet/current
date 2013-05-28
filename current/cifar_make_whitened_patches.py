"""
This script whitens CIFAR-10 images, and makes a dataset of (patch-pairs, displacement vectors) 
from the results 

"""

from pylearn2.utils import serial
from pylearn2.datasets import preprocessing
from pylearn2.utils import string_utils
import numpy as np
from pylearn2.datasets.cifar10 import CIFAR10

data_dir = string_utils.preprocess('/u/kruegers/repo/current/pylearn2/pylearn2/datasets/cifar10')

print 'Loading CIFAR-10 train dataset...'
train = CIFAR10(which_set = 'train')

print "Preparing output directory..."
output_dir = data_dir
serial.mkdir( output_dir )





print "Learning the preprocessor and preprocessing the unsupervised train data..."
preprocessor = preprocessing.ZCA()
train.apply_preprocessor(preprocessor = preprocessor, can_fit = True)

print 'Saving the unsupervised data'
train.use_design_loc(output_dir+'/train.npy')
serial.save(output_dir + '/train.pkl', train)






print "Loading the test data"
test = CIFAR10(which_set = 'test')

print "Preprocessing the test data"
test.apply_preprocessor(preprocessor = preprocessor, can_fit = False)

print "Saving the test data"
test.use_design_loc(output_dir+'/test.npy')
serial.save(output_dir+'/test.pkl', test)

serial.save(output_dir + '/preprocessor.pkl',preprocessor)

