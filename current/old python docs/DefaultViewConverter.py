class DefaultViewConverter(object):
    def __init__(self, shape, axes = ('b', 0, 1, 'c')):
        self.shape = shape
        self.pixels_per_channel = 1
        for dim in self.shape[:-1]:
            self.pixels_per_channel *= dim
        self.axes = axes

    def view_shape(self):
        return self.shape

    def weights_view_shape(self):
        return self.shape

    def design_mat_to_topo_view(self, X):
        assert len(X.shape) == 2
        batch_size = X.shape[0]

        channel_shape = [batch_size, self.shape[0], self.shape[1], 1]
        dimshuffle_args = [('b', 0, 1, 'c').index(axis) for axis in self.axes]
        if self.shape[-1] * self.pixels_per_channel != X.shape[1]:
            raise ValueError('View converter with ' + str(self.shape[-1]) +
                             ' channels and ' + str(self.pixels_per_channel) +
                             ' pixels per channel asked to convert design'
                             ' matrix with ' + str(X.shape[1]) + ' columns.')
        start = lambda i: self.pixels_per_channel * i
        stop = lambda i: self.pixels_per_channel * (i + 1)
        channels = [X[:, start(i):stop(i)].reshape(*channel_shape).transpose(*dimshuffle_args)
                    for i in xrange(self.shape[-1])]

        channel_idx = self.axes.index('c')
        rval = np.concatenate(channels, axis=channel_idx)
        assert len(rval.shape) == len(self.shape) + 1
        return rval

    def design_mat_to_weights_view(self, X):
        rval =  self.design_mat_to_topo_view(X)

        # weights view is always for display
        rval = np.transpose(rval, tuple(self.axes.index(axis)
            for axis in ('b', 0, 1, 'c')))

        return rval

    def topo_view_to_design_mat(self, V):

        # Does this line actually do anything?  
        V = V.transpose(self.axes.index('b'), self.axes.index(0),
                self.axes.index(1), self.axes.index('c'))

        num_channels = self.shape[-1]
        # Examine HERE
        if N.any(N.asarray(self.shape) != N.asarray(V.shape[1:])):
            raise ValueError('View converter for views of shape batch size '
                             'followed by ' + str(self.shape) +
                             ' given tensor of shape ' + str(V.shape))
        batch_size = V.shape[0]

        #
        rval = N.zeros((batch_size, self.pixels_per_channel * num_channels),
                       dtype=V.dtype)

        for i in xrange(num_channels):
            ppc = self.pixels_per_channel
            rval[:, i * ppc:(i + 1) * ppc] = V[..., i].reshape(batch_size, ppc)
        assert rval.dtype == V.dtype

        return rval

    def __setstate__(self, d):
        # Patch old pickle files that don't have the axes attribute.
        if 'axes' not in d:
            d['axes'] = ['b', 0, 1, 'c']
        self.__dict__.update(d)

