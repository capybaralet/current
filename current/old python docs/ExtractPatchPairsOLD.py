class ExtractPatchPairs(Preprocessor):

    """
    EDIT: new strategy - making an ExtractPatchesWithPosition class, and a MakePatchPairs class

    Default will be random patch pairs with a global grid encoding of displacements

    Going with Option 2 for now: 	
	   - choose # of patches
	   - generate patches 
	   - use all resulting pairs
	   - OPTIONAL: no_overlap (NOT IMPLEMENTED!)
    """

    def __init__(self, patch_shape, num_patches, no_overlap = False, rng=None):
        # Note: patch_shape does NOT include a channel dimension
        self.patch_shape = patch_shape
        self.num_patches = num_patches
        self.no_overlap = no_overlap

        if rng != None:
            self.start_rng = copy.copy(rng)
        else:
            self.start_rng = np.random.RandomState([1, 2, 3])

    def apply(self, dataset, can_fit=False):
        rng = copy.copy(self.start_rng)

        X = dataset.get_topological_view()

        num_topological_dimensions = len(X.shape) - 2

        if num_topological_dimensions != len(self.patch_shape):
            raise ValueError("ExtractPatches with "
                             + str(len(self.patch_shape))
                             + " topological dimensions called on "
                             + "dataset with "
                             + str(num_topological_dimensions) + ".")

        # make a "bag of patches" that we will draw pairs from
        # patches will retain position info so we can calculate displacement
        patches_shape = [self.num_patches]
        # Using all patch pairs gives this size output
        output_shape = [self.num_patches * (self.num_patches-1)]
        # topological dimensions
        for dim in self.patch_shape:
            patches_shape.append(dim)
            output_shape.append(dim)
        # number of channels
        patches_shape.append(X.shape[-1])
        output_shape.append(X.shape[-1])

        # EDIT HERE to have displacement data
        patches = np.zeros(patches_shape, dtype = X.dtype)
        output = np.zeros(output_shape, dtype=X.dtype)

        channel_slice = slice(0, X.shape[-1])
        for i in xrange(self.num_patches):
            args = []
            args.append(rng.randint(X.shape[0]))

            for j in xrange(num_topological_dimensions):
                max_coord = X.shape[j + 1] - self.patch_shape[j]
                coord = rng.randint(max_coord + 1)
                dim_slice = slice(coord, coord + self.patch_shape[j])
                args.append(dim_slice)
            args.append(channel_slice)

            patches[i, :] = X[args]

        # EDIT HERE new function probably needed
        dataset.set_topological_view(output)
        dataset.y = None













#############################################
# OLD
























    def __init__(self, num_patches, patch_stride, grid_shape, no_overlap = False):
        self.patch_shape = patch_shape
        self.patch_stride = patch_stride
        self.grid_shape = grid_shape

    def apply(self, dataset, can_fit=False):
        X = dataset.get_topological_view()
        num_topological_dimensions = len(X.shape) - 2
        if num_topological_dimensions != len(self.patch_shape):
            raise ValueError("ExtractGridPatches with "
                             + str(len(self.patch_shape))
                             + " topological dimensions called on"
                             + " dataset with " +
                             str(num_topological_dimensions) + ".")
        num_patches = X.shape[0]
        max_strides = [X.shape[0] - 1]
        for i in xrange(num_topological_dimensions): 
            patch_width = self.patch_shape[i]
            #panel_width = patch_width*panel_shape[i]
            data_width = X.shape[i + 1]
            #last_valid_coord = data_width - panel_width
            if last_valid_coord < 0:
                raise ValueError('On topological dimension ' + str(i) +
                                 ', the data has width ' + str(data_width) +
                                 ' but the requested total panel width is ' +
                                 str(patch_width) + ' * ' str(panel_shape[i])
                                 + ' = ' + str(panel_width))
            stride = self.patch_stride[i]
            if stride == 0:
                max_stride_this_axis = 0
            else:
                max_stride_this_axis = last_valid_coord / stride
            num_strides_this_axis = max_stride_this_axis + 1
            max_strides.append(max_stride_this_axis)
            num_patches *= num_strides_this_axis
        # batch size
        output_shape = [num_patches]
        # topological dimensions
        for dim in self.patch_shape:
            output_shape.append(dim)
        # number of channels
        output_shape.append(X.shape[-1])
        output = np.zeros(output_shape, dtype=X.dtype)
        channel_slice = slice(0, X.shape[-1])
        coords = [0] * (num_topological_dimensions + 1)
        keep_going = True
        i = 0
        while keep_going:
            args = [coords[0]]
            for j in xrange(num_topological_dimensions):
                coord = coords[j + 1] * self.patch_stride[j]
                args.append(slice(coord, coord + self.patch_shape[j]))
            args.append(channel_slice)
            patch = X[args]
            output[i, :] = patch
            i += 1
            # increment coordinates
            j = 0
            keep_going = False
            while not keep_going:
                if coords[-(j + 1)] < max_strides[-(j + 1)]:
                    coords[-(j + 1)] += 1
                    keep_going = True
                else:
                    coords[-(j + 1)] = 0
                    if j == num_topological_dimensions:
                        break
                    j = j + 1
        dataset.set_topological_view(output)
