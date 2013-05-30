    def topo_view_to_design_mat(self, V):

        # Does this do anything? (I think not, at least in our case)
        V = V.transpose(self.axes.index('b'), self.axes.index(0),
                self.axes.index(1), self.axes.index('c'))

        num_channels = self.shape[-1]
        if N.any(N.asarray(self.shape) != N.asarray(V.shape[1:])):
            raise ValueError('View converter for views of shape batch size '
                             'followed by ' + str(self.shape) +
                             ' given tensor of shape ' + str(V.shape))
        # equals number of patches in our case.  
        batch_size = V.shape[0]

        rval = N.zeros((batch_size, self.pixels_per_channel * num_channels),
                       dtype=V.dtype)

        for i in xrange(num_channels):
            ppc = self.pixels_per_channel
            rval[:, i * ppc:(i + 1) * ppc] = V[..., i].reshape(batch_size, ppc)

        assert rval.dtype == V.dtype

        return rval

