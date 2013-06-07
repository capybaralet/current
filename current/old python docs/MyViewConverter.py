class MyViewConverter(DefaultViewConverter):
    # This view converter allows the user to create a new dataset with a flat encoding
    # X[example_num.:] = example_data is a 1d array.  This new encoding does not easily  
    # translate back into the old design matrix view, or topo view, so we add a parameter 
    # to save the old design matrix view (old_X) (in the getter...), 
    # and (TODO) modify methods accordingly
    # 
    # We use it to encode patch-pairs with displacement as training examples.  
    #
    # We should make our usage a subclass, and include options no_overlap to prevent 
    # including pairs that share pixels, and no_reflection (or another name) to avoid 
    # including both (p1,p2,d) and (p2,p1,-d) 
    def __init__(self, shape, axes = ('b', 0, 1, 'c')):
        self.shape = shape
        self.pixels_per_channel = 1
        for dim in self.shape[:-1]:
            self.pixels_per_channel *= dim
        self.axes = axes

    # rename max_stamp
    def flatten_encoding(encoding, max_stamp):
        dims = len(encoding)
        flat_encoding = 0
        for i in xrange(dims-1):
             flat_encoding += encoding[i]
             flat_encoding *= max_stamp
        flat_encoding += encoding[-1]


    # this constructs the augmented view based on the datasets current
    # dense_design_matrix, here called X, and stamps.   
    def convert_design_mat_augmented(self, X):

        # code below is pasted from run.py
        # we need to get parameters to make it work.
        # I'm giving up on the view converter for now, and going 
        # back to doing everything (left) within run.py
        num_images
        patches_per_image


        for i in xrange(num_images):
            for j in xrange(patches_per_image):
                patch1_num = i * patches_per_image + j
                patch1_pos = stamps[patch1_num,:]
                for k in xrange(patches_per_image):
                    example_num = i*examples_per_image + j*(patches_per_image-1) + k
                    if (k > j):
                        example_num -= 1
                    if (k != j):                    
                        patch2_num = i * patches_per_image + k
                        patch2_pos = stamps[patch2_num,:]
                        distance = patch1_pos - patch2_pos
                        distances[example_num] = distance
                        distance_encoding = distance + max_stamp
                        distance_encoding = flatten_encoding(distance_encoding)
                        distances_onehot[example_num, distance_encoding] = 1
                        patch_pairs[example_num] = np.concatenate(design_matrix[patch1_num], design_matrix[patch2_num])
                        examples[example_num] = np.concatenate(patch_pairs[example_num], distances_onehot[example_num]) 




        
        



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
