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
