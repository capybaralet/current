def set_topological_view(self, V, axes = ('b', 0, 1, 'c'), start = 0):
    """
    Sets the dataset to represent V, where V is a batch
    of topological views of examples.

    Parameters
    ----------
    V : ndarray
        An array containing a design matrix representation of training
        examples. If unspecified, the entire dataset (`self.X`) is used
        instead.
    TODO: why is this parameter named 'V'?
    """
    assert not N.any(N.isnan(V))
    rows = V.shape[axes.index(0)]
    cols = V.shape[axes.index(1)]
    channels = V.shape[axes.index('c')]
    self.view_converter = DefaultViewConverter([rows, cols, channels], axes=axes)
    X = self.view_converter.topo_view_to_design_mat(V)
    assert not N.any(N.isnan(X))
    DenseDesignMatrixPyTables.fill_hdf5(file = self.h5file,
                                        data_x = X,
                                        start = start)
