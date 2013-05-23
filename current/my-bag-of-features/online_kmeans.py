import numpy
import numpy.random


def kmeans(X, K, numepochs, Winit=None, learningrate=0.01, batchsize=50, verbose=True): 
    if Winit is None:
        W = numpy.random.randn(K, X.shape[1])*0.1
    else:
        W = Winit

    X2 = (X**2).sum(1)[:, None]
    for epoch in range(numepochs):
        for i in range(0, X.shape[0], batchsize):
            W2 = (W**2).sum(1)[:, None]
            D = -2*numpy.dot(W, X[i:i+batchsize,:].T) + W2 + X2[i:i+batchsize].T
            S = (D==D.min(0)[None,:]).astype("float").T
            clustersums = numpy.dot(S.T, X[i:i+batchsize,:])
            pointspercluster = S.sum(0)[:, None]
            W += learningrate * (clustersums - pointspercluster * W) 
        if verbose:
            cost = D.min(0).sum()
            print "epoch", epoch+1, "of", numepochs, " cost: ", cost

    return W


def assign(X, W):
    X2 = (X**2).sum(1)[:, None]
    W2 = (W**2).sum(1)[:, None]
    D = -2*numpy.dot(W, X.T) + W2 + X2.T
    return (D==D.min(0)[None,:]).astype(int)


def assign_triangle(X, W):
    X2 = (X**2).sum(1)[:, None]
    W2 = (W**2).sum(1)[:, None]
    D = -2*numpy.dot(W, X.T) + W2 + X2.T
    D = (D.mean(0)[None,:]-D)
    return (D*(D>0)).T # Is this normalized?  Should it be?


