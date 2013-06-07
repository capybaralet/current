import numpy as np
import numpy.random
import pylab
import theano 
import pylearn2
import pylearn2.models.autoencoder as autoencoder

from pylearn2.config import yaml_parse
from pylearn2.utils import serial

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm



###
# Note: Currently training is failing dramatically



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
learningrate = .1



def s(x):
    return 1. / (1. + np.exp(-x)) 

model = serial.load('run.pkl')
dataset_yaml_src = model.dataset_yaml_src
dataset = yaml_parse.load(dataset_yaml_src)

w1 = model.weights
w2 = model.w_prime
b1 = model.hidbias
b2 = model.visbias
w1 = w1.get_value()
w2 = w2.get_value()
b1 = b1.get_value()
b2 = b2.get_value()

# Plotting this many example input/output pairs
num2plot = 8

inputs = dataset.X
flat_inputs = inputs
h = s(np.dot(flat_inputs,w1)+b1)
flat_outputs = s(np.dot(h,w2)+b2)
outputs = flat_outputs


