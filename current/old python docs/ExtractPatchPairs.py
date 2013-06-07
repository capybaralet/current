class ExtractPatchPairs(Preprocessor):
    """
    Right now, this assumes the dataset has just been preprocessed
    by ExtractPatchesWithPosition (EPWP).  I left the rng stuff
    and maybe other bits that we will want if we change it to
    automatically add EPWP to the pipeline.
    """
    def __init__(self, patches_per_image, num_images, input_width, rng=None):
        self.patches_per_image = patches_per_image
        # should have num_patches attribute?
        # self.num_patches = num_patches
        self.num_images = num_images
        self.input_width = input_width

        if rng != None:
            self.start_rng = copy.copy(rng)
        else:
            self.start_rng = np.random.RandomState([1, 2, 3])

    def apply(self, dataset, can_fit=False):
        rng = copy.copy(self.start_rng)

        num_images = self.num_images
        
        patches_per_image = self.patches_per_image
        examples_per_image = patches_per_image * (patches_per_image-1)

        stamps = dataset.stamps
        topo_view = dataset.get_topological_view()
        design_matrix = dataset.get_design_matrix()
        processed_patch_size = design_matrix.shape[1]

        # Here again, we are assuming 2D images
        input_dim = 2
        input_width = self.input_width
        patch_width = topo_view.shape[1]
        num_examples = examples_per_image * num_images

        max_stamp = input_width - patch_width
        d_size = (2*max_stamp+1)**input_dim

        patch_pairs = np.zeros((num_examples, 2*processed_patch_size))
        distances = np.zeros((num_examples, input_dim))
        distances_onehot = np.zeros((num_examples, d_size))
        examples = np.zeros((num_examples, 2*processed_patch_size + d_size))

        nvis = 2*processed_patch_size + d_size


        def flatten_encoding(encoding, max_stamp):
            dims = len(encoding)
            flat_encoding = 0
            for i in xrange(dims-1):
                 flat_encoding += encoding[i]
                 flat_encoding *= max_stamp
            flat_encoding += encoding[-1]


        # Can be done without (or with less) for loops?
        print 'begin for loop'
        for i in xrange(num_images):
            if (i%3000 == 0):
                print i, '-th image being processed...'
            for j in xrange(patches_per_image):
                patch1_num = i * patches_per_image + j
                patch1_pos = stamps[patch1_num,:]
                for k in xrange(patches_per_image):
                    example_num = i*examples_per_image + \
                                  j*(patches_per_image-1) + k
                    if (k > j):
                        example_num -= 1
                    if (k != j):
                        patch2_num = i * patches_per_image + k
                        patch2_pos = stamps[patch2_num,:]
                        distance = patch1_pos - patch2_pos
                        distances[example_num] = distance
                        distance_encoding = distance + max_stamp
                        distance_encoding = flatten_encoding(distance_encoding,
                                                             max_stamp
                                                            )
                        distances_onehot[example_num, distance_encoding] = 1
                        p1 = design_matrix[patch1_num]
                        p2 = design_matrix[patch2_num]
                        patch_pairs[example_num] = np.hstack((p1, p2))
                        examples[example_num] = np.hstack(
                                                    (patch_pairs[example_num],
                                                     distances_onehot[example_num])
                                                         )

        dataset.set_design_matrix(examples)