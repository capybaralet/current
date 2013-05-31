# This is a script to train Denoising Autoenconders with a range of hyper-parameters
# on patch-pair data extracted from images, save the trained models, and plot example 
# input/reconstruction pairs.
# 
# Currently, we are using CIFAR-10 for the datset, but we will change this to using face data.
#
# This script will also likely be broken up into several scripts for modularity once I have it 
# running smoothly. 
#
# 

import numpy 
import numpy.random

import matplotlib
import matplotlib.pyplot as plt

import pylab
import theano 
import pylearn2
import pylearn2.models.autoencoder as autoencoder

from pylearn2.config import yaml_parse


#Step 0: Set Parameters

# Default to 3 for RGB
numchannels = 3
# We assume a square input of inputdim x inputdim x numchannels
inputdim = 32

nclasses = 10
irange = .5


nhid = [100, 200, 300]
patchsize = [4, 6, 9]
#pooling = [1, 2, 4]
pooling = 1
stdev  = [.3, 1, 3]
#dweight = [30, 90]
dweight = 50



#Step 1: Generate and save Models



#Step 2: Plot Results
