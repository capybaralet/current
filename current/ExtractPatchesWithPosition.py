class ExtractPatchesWithPosition(Preprocessor):
        """ 
        The original ExtractPatches class converts an 
        image dataset into a dataset of patches
        extracted at random from the original dataset. 

        This class does the same, but will also keep 
        data about each patch's location

        Crap, just realized another issue with this:
        It needs to know what image it came from as well.

        I'm now looking into Blocks, because it looks like 
        the best idea is maybe to process one image at a time.

        OK, Blocks seem outdated/about layers of the network, 
        not preprocessing.... basically, prolly not the way to go.

        I'm going to look at ExtractGridPatches (and companion)
        again, since they seem to keep track of things somehow.
        """
    def __init__(self, patch_shape, num_patches, rng=None):
        self.patch_shape = patch_shape
        self.num_patches = num_patches

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

        # batch size
        output_shape = [self.num_patches]
        # topological dimensions
        for dim in self.patch_shape:
            output_shape.append(dim)
        # number of channels
        output_shape.append(X.shape[-1])
        output = np.zeros(output_shape, dtype=X.dtype)
        channel_slice = slice(0, X.shape[-1])
        for i in xrange(self.num_patches):
            args = []
            # EDIT HERE: this picks a random image for each patch.
            # We want to go through each image sequentially and 
            # take the same number of patches for each image
            args.append(rng.randint(X.shape[0]))

            for j in xrange(num_topological_dimensions):
                max_coord = X.shape[j + 1] - self.patch_shape[j]
                coord = rng.randint(max_coord + 1)
                args.append(slice(coord, coord + self.patch_shape[j]))
            args.append(channel_slice)
            output[i, :] = X[args]
        dataset.set_topological_view(output)
        dataset.y = None

