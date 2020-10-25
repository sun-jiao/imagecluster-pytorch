#!/usr/bin/python3

from matplotlib import pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram

from imagecluster import calc, io


if __name__ == '__main__':
    images = io.read_images('stellera/', size=(224, 224))
    model = calc.get_model()
    fingerprints = calc.fingerprints(images, model)
    clusters, extra = calc.cluster(fingerprints, sim=0.5, extra_out=True)

    # linkage matrix Z
    fig, ax = plt.subplots(figsize = (20, 40))
    dendrogram(extra['Z'], ax=ax, orientation='left', labels=extra['labels'])

    # Adjust yaxis labels (values from Z[:,2]) to our definition of the `sim`
    # parameter.
    xmin, xmax = ax.xaxis.get_data_interval()
    tlocs = np.linspace(xmin, xmax, 5)
    ax.xaxis.set_ticks(tlocs)
    tlabels = np.linspace(1, 0, len(tlocs))
    ax.xaxis.set_ticklabels(tlabels)
    ax.set_ylabel("filename")
    ax.set_xlabel("similarity")
    ax.tick_params(axis='x', which='major', labelsize=12)
    ax.tick_params(axis='y', which='major', labelsize=12)

    fig.savefig('dendrogram.png')
    plt.show()
