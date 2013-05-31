class ExtractPatchesWithPosition(Preprocessor):
        """ 
        Right now this only works for 2D inputs, such as images.
        This can be generalized without too much work, but it may
        require editing more other parts of pylearn2

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
    def __init__(self, patch_shape, patches_per_image, rng=None): #, input_dim = 2):
        self.patch_shape = patch_shape
        self.patches_per_image = patches_per_image
        # self.input_dim = input_dim
        # should have num_patches attribute?
        # self.num_patches = num_patches

        if rng != None:
            self.start_rng = copy.copy(rng)
        else:
            self.start_rng = np.random.RandomState([1, 2, 3])

    def apply(self, dataset, can_fit=False):
        rng = copy.copy(self.start_rng)

        X = dataset.get_topological_view()
        num_images = X.shape[0] 

        num_patches = num_images * patches_per_image

        num_topological_dimensions = len(X.shape) - 2

        if num_topological_dimensions != len(self.patch_shape):
            raise ValueError("ExtractPatches with "
                             + str(len(self.patch_shape))
                             + " topological dimensions called on "
                             + "dataset with "
                             + str(num_topological_dimensions) + ".")

        # output1 = patches, output2 = positions
        output1_shape = [self.num_patches]
        output2_shape = [self.num_patches, num_topological_dimensions]
        # topological dimensions
        for dim in self.patch_shape:
            output1_shape.append(dim)
        # number of channels
        output1_shape.append(X.shape[-1])
        output1 = np.zeros(output1_shape, dtype=X.dtype)
        channel_slice = slice(0, X.shape[-1])
         
        # We go through each image sequentially and 
        # take the same number of patches for each image.
        # 
        # We also keep track of the patches' positions in output2
        for i in xrange(self.num_images):
            for j in xrange(self.patches_per_image)
                patch_num = i * patches_per_image + j
                args = [i]
                for k in xrange(num_topological_dimensions):
                    max_coord = X.shape[k + 1] - self.patch_shape[k]
                    coord = rng.randint(max_coord + 1)
                    output2[patch_num, k] = coord
                    args.append(slice(coord, coord + self.patch_shape[k]))
                args.append(channel_slice)
                output1[patch_num, :] = X[args]
        
        dataset.set_topological_view(output1)
        dataset.y = None
        # Should have a set_tags method?
        dataset.tags = output2
