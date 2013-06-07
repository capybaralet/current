class DenseDesignMatrixAugmented(DenseDesignMatrix):
    """
        Parameters
        ----------

        X : ndarray, 2-dimensional, optional
            Should be supplied if `topo_view` is not. A design
            matrix of shape (number examples, number features)
            that defines the dataset.
        topo_view : ndarray, optional
            Should be supplied if X is not.  An array whose first
            dimension is of length number examples. The remaining
            dimensions are xamples with topological significance,
            e.g. for images the remaining axes are rows, columns,
            and channels.
        y : ndarray, 1-dimensional(?), optional
            Labels or targets for each example. The semantics here
            are not quite nailed down for this yet.
        stamps: ndarray, optional
            First dimension is the number of examples, other dimensions 
            contain extra information about the examples.  The extra 
            data can be used to construct an augmented dataset, which 
            if then stored as self.X.
            Currently, we keep track of patch position information 
            here for randomly cropped patches, and use this information
            to construct an augmented dataset with examples consisting
            of patch-pairs with displacement: (p1, p2, d).
        X_old: when we convert X to the augmented representation, we 
            save the old X here.
        X_is_augmented: when true, X contains the training examples 
            (in a 2D format), whereas X_old contains the unaugmented
            data representation (as a design matrix)
        view_converter : object, optional
            An object for converting between design matrices and
            topological views. 
        rng : object, optional
            A random number generator used for picking random
            indices into the design matrix when choosing minibatches.
    """
    ########################################################
    # OLD FUNCTIONS to EDIT

    # EDIT HERE    
    def __init__(self, X=None, topo_view=None, y=None, view_converter=None, 
                 stamps=None, X_old = None, X_is_augmented = False, 
                 axes = ('b', 0, 1, 'c'), rng=_default_seed):

        super(DenseDesignMatrixAugmented, self).__init__(X = X,
                                            topo_view = topo_view,
                                            y = y,
                                            stamps = stamps,
                                            view_converter = view_converter,
                                            axes = axes,
                                            rng = rng)

        self.X_is_augmented = X_is_augmented
        # self.view_converter = MyViewConverter(self.X)

    # Warning
    def get_topo_batch_axis(self):
        warnings.warn("Warning: for this dataset class, " 
                      "DenseDesignMatrixAugmented, "
                      "the topological and design_matrix (X) views "
                      "may not be in sync.  For more info, type "
                      "help(DenseDesignMatrixAugmented)."
        return self.view_converter.axes.index('b')

    # Warning 
    def apply_preprocessor(self, preprocessor, can_fit=False):

    # Warning
    def get_topological_view(self, mat=None):
        """
        Convert an array (or the entire dataset) to a topological view.

        Parameters
        ----------
        mat : ndarray, 2-dimensional, optional
            An array containing a design matrix representation of training
            examples. If unspecified, the entire dataset (`self.X`) is used
            instead.

            This parameter is not named X because X is generally used to
            refer to the design matrix for the current problem. In this
            case we want to make it clear that `mat` need not be the design
            matrix defining the dataset.
        """
        if self.view_converter is None:
            raise Exception("Tried to call get_topological_view on a dataset "
                            "that has no view converter")
        if mat is None:
            if self.X_old is None:
                mat = self.X
            else:
                mat = self.X_old
        return self.view_converter.design_mat_to_topo_view(mat)

    # Warning
    def get_weights_view(self, mat):



    def set_topological_view(self, V, axes = ('b', 0, 1, 'c')):

    def get_design_matrix(self, topo=None):
    
    # Warning
    def set_design_matrix(self, X):

    def get_targets(self):

    ########################################################
    # OLD FUNCTIONS to EDIT (not parsed)

    def num_examples(self):
        return self.X.shape[0]

    def get_batch_design(self, batch_size, include_labels=False):
        try:
            idx = self.rng.randint(self.X.shape[0] - batch_size + 1)
        except ValueError:
            if batch_size > self.X.shape[0]:
                raise ValueError("Requested "+str(batch_size)+" examples"
                    "from a dataset containing only "+str(self.X.shape[0]))
            raise
        rx = self.X[idx:idx + batch_size, :]
        if include_labels:
            if self.y is None:
                return rx, None
            ry = self.y[idx:idx + batch_size]
            return rx, ry
        rx = np.cast[config.floatX](rx)
        return rx

    def get_batch_topo(self, batch_size, include_labels = False):

        if include_labels:
            batch_design, labels = self.get_batch_design(batch_size, True)
        else:
            batch_design = self.get_batch_design(batch_size)

        rval = self.view_converter.design_mat_to_topo_view(batch_design)

        if include_labels:
            return rval, labels

        return rval

    def view_shape(self):
        return self.view_converter.view_shape()

    def weights_view_shape(self):
        return self.view_converter.weights_view_shape()

    def restrict(self, start, stop):
        """
        Restricts the dataset to include only the examples
        in range(start, stop). Ignored if both arguments are None.
        """
        assert (start is None) == (stop is None)
        if start is None:
            return
        assert start >= 0
        assert stop > start
        assert stop <= self.X.shape[0]
        assert self.X.shape[0] == self.y.shape[0]
        self.X = self.X[start:stop, :]
        if self.y is not None:
            self.y = self.y[start:stop, :]
        assert self.X.shape[0] == self.y.shape[0]
        assert self.X.shape[0] == stop - start


    def adjust_for_viewer(self, X):
        return X / np.abs(X).max()

    def adjust_to_be_viewed_with(self, X, ref, per_example=None):
        if per_example is not None:
            warnings.warn("ignoring per_example")
        return np.clip(X / np.abs(ref).max(), -1., 1.)








    ########################################################
    # OUR NEW FUNCTIONS

    def return_augmented(self):
        """
        Return augmented dataset, constructing it if necessary.

        """
        if self.view_converter is None:
            raise Exception("Tried to convert to augmented view using a "
                            "dataset that has no view converter"
        elif self.view_converter not isinstance(MyViewConverter):
            raise Exception("Tried to convert to augmented view using a "
                            "view converter that doesn't implement "
                            "MyViewConverter"
        elif self.is_augmented:
            return self.X
        else: 
            self.X_old = self.X
            self.X = self.view_converter.convert_design_mat_augmented(self.X)
            self.is_augmented = True
            return self.X





