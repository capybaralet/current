class MakePatchPairsWithDisplacements(Preprocessor):
        """ 
        This should convert patches (with position, and implicit knowledge of origin
        image) into examples consisting of patch pairs and displacements, in their 
        input encoding (for now this is pixel distance x, pixel distance y, 
        next stage is one-hot and then linear combo bins as per Yoshua)

        After processing, there is no good topo representation, so we set it to None 
        (for now....... we prolly need to subclass dense_design_matrix)
        """
    def __init__(self, patches_per_image):
        self.patches_per_image = patches_per_image


    def apply(self, dataset, can_fit=False):

        X = dataset.get_topological_view()
        num_patches = X.shape[0] 
        # check that there is no remainder here! (or just pass more data along)
        num_images = num_patches / patches_per_image

        patches_per_image = self.patches_per_image
	tags = self.tags

        num_topological_dimensions = len(X.shape) - 2

        if num_topological_dimensions != len(self.patch_shape):
            raise ValueError("ExtractPatches with "
                             + str(len(self.patch_shape))
                             + " topological dimensions called on "
                             + "dataset with "
                             + str(num_topological_dimensions) + ".")


        num_examples = patches_per_image * (patches_per_image-1) * num_images
        distances = [num_examples, num_topological_dimensions]

        for i in xrange(num_images):
            for j in xrange(patches_per_image):
                patch1_num = i * patches_per_image + j
                patch1_pos = tags[patch1_num,:]
                for k in xrange(patches_per_image):
                    if (j != k):                    
                        patch2_num = i * patches_per_image + k
                        patch2_pos = tags[patch2_num,:]
                        distances[
















        # output1 = patches, output2 = positions
        output1_shape = [num_patches]
        output2_shape = [num_patches, num_topological_dimensions]
        # topological dimensions
        for dim in self.patch_shape:
            output1_shape.append(dim)
        # number of channels
        output1_shape.append(X.shape[-1])
        output1 = np.zeros(output1_shape, dtype=X.dtype)
        output2 = np.zeros(output2_shape, dtype=X.dtype)        
        channel_slice = slice(0, X.shape[-1])
         
        # We go through each image sequentially and 
        # take the same number of patches for each image.
        # 
        # We also keep track of the patches' positions in output2
        for i in xrange(num_images):
            for j in xrange(patches_per_image):
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

