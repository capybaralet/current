


import numpy
from numpy.linalg import eigh




def pca(data, var_fraction):
    """ principal components, retaining as many components as required to
retain var_fraction of the variance

Returns projected data, projection mapping, inverse mapping, mean"""


    u, v = eigh(numpy.cov(data, rowvar=1, bias=1))
    v = v[:, numpy.argsort(u)[::-1]]
    u.sort()
    u = u[::-1]
    u = u[u.cumsum()<u.sum()*var_fraction]
    numprincomps = u.shape[0]
    V = ((u**(-0.5))[:numprincomps][numpy.newaxis,:]*v[:,:numprincomps]).T
    W = (u**0.5)[:numprincomps][numpy.newaxis,:]*v[:,:numprincomps]
    return V, W
    #return numpy.dot(V,data), V, W
