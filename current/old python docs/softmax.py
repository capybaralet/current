import numpy 
import numpy.random
import pylab
import theano 
import pylearn2

from pylearn2.config import yaml_parse


dataset = """!obj:pylearn2.datasets.mnist.MNIST {
        which_set: 'train',
        one_hot: 1,
        start: 0,
        stop: 50000
    }"""
        
model = """!obj:pylearn2.models.softmax_regression.SoftmaxRegression {
    n_classes: 10,
    irange: 0.,
    nvis: 784,
}"""

algorithm = """!obj:pylearn2.training_algorithms.bgd.BGD {
        batch_size: 10000,
        line_search_mode: 'exhaustive',
        conjugate: 1,
        monitoring_dataset:
            {
                'train' : *train,
                'valid' : !obj:pylearn2.datasets.mnist.MNIST {
                              which_set: 'train',
                              one_hot: 1,
                              start: 50000,
                              stop:  60000
                          },
                'test'  : !obj:pylearn2.datasets.mnist.MNIST {
                              which_set: 'test',
                              one_hot: 1,
                          }
            },
        termination_criterion: !obj:pylearn2.termination_criteria.MonitorBased {
            channel_name: "valid_y_misclass"
        }
    }"""
    
train = """!obj:pylearn2.train.Train {
    dataset: &train %(dataset)s,
    model: %(model)s,
    algorithm: %(algorithm)s,
    extensions: [
        !obj:pylearn2.train_extensions.best_params.MonitorBasedSaveBest {
             channel_name: 'valid_y_misclass',
             save_path: "softmax_regression_best.pkl"
        },
    ],
    save_path: "softmax_regression.pkl",
    save_freq: 1
}""" % locals()


train = yaml_parse.load(train)
train.main_loop()