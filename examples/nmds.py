#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance
from sklearn.manifold import MDS

from imagecluster import calc, io


def fingerprints_to_distance_matrix(fingerprints):
    """
    Converts the fingerprint dictionary into a symmetric distance matrix.

    Parameters
    ----------
    fingerprints : dict
        A dictionary of feature vectors output by the `fingerprints` function.

    Returns
    -------
    dist_matrix : 2D array
        A symmetric distance matrix for NMDS analysis.
    filenames : list
        A list of filenames corresponding to the distance matrix.
    """
    _filenames = list(fingerprints.keys())
    _feature_vectors = np.array(list(fingerprints.values()))

    _dist_matrix = distance.squareform(distance.pdist(_feature_vectors, metric='euclidean'))

    return _dist_matrix, _filenames


if __name__ == '__main__':
    images = io.read_images('pics/', size=(224, 224))
    model = calc.get_model()
    fingerprints = calc.fingerprints(images, model)
    dist_matrix, filenames = fingerprints_to_distance_matrix(fingerprints)

    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    results = mds.fit_transform(dist_matrix)

    x = results[:, 0]
    y = results[:, 1]

    fig, ax = plt.subplots()

    ax.scatter(x, y)

    for i, label in enumerate(filenames):
        ax.text(x[i], y[i], label, fontsize=12, ha='center', va='center')

    plt.show()