"""
This script makes a dataset of two million approximately whitened patches, extracted at random uniformly
from a downsampled version of the STL-10 unlabeled and train dataset.

It assumes that you have already run make_downsampled_stl10.py, which downsamples the STL-10 images to
1/3 of their original resolution.

This script is intended to reproduce the preprocessing used by Adam Coates et. al. in their work from
the first half of 2011.
"""

from pylearn2.utils import serial
from pylearn2.datasets import preprocessing
from pylearn2.utils import string
import numpy as np






"""
Makes a version of the STL-10 dataset that has been downsampled by a factor of
3 along both axes.
"""

from pylearn2.datasets.cifar10 import CIFAR10
from pylearn2.utils import string_utils as string

print 'Preparing output directory...'

data_dir = string.preprocess('/u/kruegerd/repo/current/pylearn2/pylearn2/datasets/cifar10')
downsampled_dir = data_dir
serial.mkdir( downsampled_dir )

#Unlabeled dataset is huge, so do it in chunks
#(After downsampling it should be small enough to work with)
final_unlabeled = np.zeros((100*1000,32*32*3),dtype='float32')

for i in xrange(10):
    print 'Loading unlabeled chunk '+str(i+1)+'/10...'
    unlabeled = STL10(which_set = 'unlabeled', center = True,
            example_range = (i * 10000, (i+1) * 10000))

    print 'Preprocessing unlabeled chunk...'
    print 'before ',(unlabeled.X.min(),unlabeled.X.max())
    unlabeled.apply_preprocessor(preprocessor)
    print 'after ',(unlabeled.X.min(), unlabeled.X.max())

    final_unlabeled[i*10000:(i+1)*10000,:] = unlabeled.X

unlabeled.set_design_matrix(final_unlabeled)
print 'Saving unlabeleding set...'
unlabeled.enable_compression()
unlabeled.use_design_loc(downsampled_dir + '/unlabeled.npy')
serial.save(downsampled_dir+'/unlabeled.pkl',unlabeled)

del unlabeled
import gc
gc.collect()

print 'Loading testing set...'
test = STL10(which_set = 'test', center = True)

print 'Preprocessing testing set...'
print 'before ',(test.X.min(),test.X.max())
test.apply_preprocessor(preprocessor)
print 'after ',(test.X.min(), test.X.max())

print 'Saving testing set...'
test.enable_compression()
test.use_design_loc(downsampled_dir + '/test.npy')
serial.save(downsampled_dir+'/test.pkl',test)
del test

print 'Loading training set...'
train = STL10(which_set = 'train', center = True)

print 'Preprocessing training set...'
print 'before ',(train.X.min(),train.X.max())
train.apply_preprocessor(preprocessor)
print 'after ',(train.X.min(), train.X.max())

print 'Saving training set...'
train.enable_compression()
train.use_design_loc(downsampled_dir + '/train.npy')
serial.save(downsampled_dir+'/train.pkl',train)

del training














data_dir = string.preprocess('/u/kruegerd/repo/current/pylearn2/pylearn2/datasets/cifar10')

print 'Loading cifar10 unlabeled and train datasets...'
downsampled_dir = data_dir

data = serial.load(downsampled_dir + '/unlabeled.pkl')
supplement = serial.load(downsampled_dir + '/train.pkl')

print 'Concatenating datasets...'
data.set_design_matrix(np.concatenate((data.X,supplement.X),axis=0))
del supplement


print "Preparing output directory..."
patch_dir = data_dir + '/stl10_patches_8x8'
serial.mkdir( patch_dir )
README = open(patch_dir + '/README','w')

README.write("""
The .pkl files in this directory may be opened in python using
cPickle, pickle, or pylearn2.serial.load.

data.pkl contains a pylearn2 Dataset object defining an unlabeled
dataset of 2 million 6x6 approximately whitened, contrast-normalized
patches drawn uniformly at random from a downsampled (to 32x32)
version of the STL-10 train and unlabeled datasets.

preprocessor.pkl contains a pylearn2 Pipeline object that was used
to extract the patches and approximately whiten / contrast normalize
them. This object is necessary when extracting features for
supervised learning or test set classification, because the
extracted features must be computed using inputs that have been
whitened with the ZCA matrix learned and stored by this Pipeline.

They were created with the pylearn2 script make_stl10_patches.py.

All other files in this directory, including this README, were
created by the same script and are necessary for the other files
to function correctly.
""")

README.close()

print "Preprocessing the data..."
pipeline = preprocessing.Pipeline()
pipeline.items.append(preprocessing.ExtractPatches(patch_shape=(8,8),num_patches=2*1000*1000))
pipeline.items.append(preprocessing.GlobalContrastNormalization(sqrt_bias=10., use_std=True))
pipeline.items.append(preprocessing.ZCA())
data.apply_preprocessor(preprocessor = pipeline, can_fit = True)

data.use_design_loc(patch_dir + '/data.npy')

serial.save(patch_dir + '/data.pkl',data)

serial.save(patch_dir + '/preprocessor.pkl',pipeline)