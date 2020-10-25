#!/usr/bin/python3

# Minimal example. Use the convenience function io.get_image_data() without any
# extra arguments.

from imagecluster import calc, io, postproc

if __name__ == '__main__':

    # The bottleneck is calc.fingerprints() called in this function, all other
    # operations are very fast. get_image_data() writes fingerprints to disk and
    # loads them again instead of re-calculating them.
    images,fingerprints,timestamps = io.get_image_data('pics/')

    # Run clustering on the fingerprints. Select clusters with similarity index
    # sim=0.5.
    clusters = calc.cluster(fingerprints, sim=0.9)

    # Create dirs with links to images. Dirs represent the clusters the images
    # belong to.
    postproc.make_links(clusters, 'clusters')

    # Plot images arranged in clusters.
    postproc.visualize(clusters, images)