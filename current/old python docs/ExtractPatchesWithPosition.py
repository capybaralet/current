class ExtractPatchesWithPosition(Preprocessor):
    """ 
    Right now this only works for 2D inputs, such as images.
    This can be generalized without too much work, but it may
    require editing more other parts of pylearn2.
    
    The original ExtractPatches class converts an 
    image dataset into a dataset of patches
    extracted at random from the original dataset. 
    
    This class does the same, but will also keep 
    data about each patch's location.  We track which 
    image a patch belonged to implicitly, by filling 
    the output array in order starting with all the patches
    from the first image.
    
    We construct another array as well: the ordered pair 
    ( patch_positions[i,1] , [patch_positions[i,2] )
    represents the patch i's (x,y) position in the image, 
    in terms starting pixels.

    Maybe this should be changed to the "bins" encoding.
    """
    def __init__(self, patch_shape, patches_per_image, rng=None):
        self.patch_shape = patch_shape
        self.patches_per_image = patches_per_image
        # should have num_patches attribute?
        # self.num_patches = num_patches

        if rng != None:
            self.start_rng = copy.copy(rng)
        else:
            self.start_rng = np.random.RandomState([1, 2, 3])

    def apply(self, dataset, can_fit=False):
        rng = copy.copy(self.start_rng)

        V = dataset.get_topological_view()
        num_images = V.shape[0]

        patches_per_image = self.patches_per_image
        num_patches = num_images * patches_per_image

        num_topological_dimensions = len(V.shape) - 2

        if num_topological_dimensions != len(self.patch_shape):
            raise ValueError("ExtractPatchesWithPosition with "
                             + str(len(self.patch_shape))
                             + " topological dimensions called on "
                             + "dataset with "
                             + str(num_topological_dimensions) + ".")

        # patches = patches, stamps = positions
        patches_shape = [num_patches]
        stamps_shape = [num_patches, num_topological_dimensions]
        # topological dimensions
        for dim in self.patch_shape:
            patches_shape.append(dim)
        # number of channels
        patches_shape.append(V.shape[-1])
        patches = np.zeros(patches_shape, dtype=V.dtype)
        stamps = np.zeros(stamps_shape, dtype=V.dtype)
        channel_slice = slice(0, V.shape[-1])

        # We go through each image sequentially and 
        # take the same number of patches for each image.
        #
        # We also keep track of the patches' positions in stamps
        for i in xrange(num_images):
            for j in xrange(patches_per_image):
                patch_num = i * patches_per_image + j
                args = [i]
                for d in xrange(num_topological_dimensions):
                    max_coord = V.shape[d + 1] - self.patch_shape[d]
                    coord = rng.randint(max_coord + 1)
                    stamps[patch_num, d] = coord
                    args.append(slice(coord, coord + self.patch_shape[d]))
                args.append(channel_slice)
                patches[patch_num, :] = V[args]

        dataset.set_topological_view(patches)
        dataset.y = None
        # Should have a set_stamps method?
        dataset.stamps = stamps
