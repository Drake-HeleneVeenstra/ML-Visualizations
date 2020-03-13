###
# This script tests different types of clustering, and compares performance
# Adapted from https://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html
# As accessed 10 March 2020
###

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
import sklearn.datasets as data
import time
import hdbscan
from joblib import Memory

sns.set_context('poster')
sns.set_style('white')
sns.set_color_codes()
plot_kwds = {'alpha': 0.5, 's': 80, 'linewidths': 0}

# Make some data
moons, _ = data.make_moons(n_samples=100, noise=0.05)
blobs, _ = data.make_blobs(n_samples=100, centers=[(1, 2.25), (-1.0, 1.9)], cluster_std=0.2)
data = np.vstack([moons, blobs])

# Scatterplot some data
f1 = plt.figure(1)
plt.scatter(data.T[0], data.T[1], c='b', **plot_kwds)
plt.title('Scatterplot of generated data')
frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)


# Cluster and plot, time performance
def plot_clusters(data_in, algorithm, args, kwds):
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data_in)
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data_in.T[0], data_in.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)),
              fontsize=24)
    plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time),
             fontsize=14)


# Test different clustering algorithms:
# K-means - partitioning algorithm assuming global partitions
# (centroid-based).
# Needs specification on number of clusters.
# Performance: poor: splices and clumps incorrectly:
f2 = plt.figure(2)
plot_clusters(data, cluster.KMeans, (), {'n_clusters': 5})

# Affinity propagation - graph based voting of points, then centroid-based
# partitioning approach. No need for specification nr of clusters.
# Performance: poor, it splices and clumps incorrectly.
# Difficult to determine preference values.
# Also slower than K-means.
f3 = plt.figure(3)
plot_clusters(data, cluster.AffinityPropagation, (),
              {'preference': -6.0, 'damping': 0.85})

# Mean shift - also centroid based but clustering rather than
# partitioning. Approach of probability density function.
# Performance: suboptimal: although less noise pollution, there
# is  improper splicing and a lack of inclusion. Also slow.
f4 = plt.figure(4)
plot_clusters(data, cluster.MeanShift, (0.3,), {'cluster_all': False})

# Spectral clustering - graph clustering, manifold learning
# (transformation of space), then K-means clustering
# Performance: suboptimal; improper splicing and pollution
f5 = plt.figure(5)
plot_clusters(data, cluster.SpectralClustering, (), {'n_clusters': 5})

# Agglomerative clustering - repeatedly merging towards larger clusters
# Dendrogram of clusters which can be broken down to clusters.
# Performance: suboptimal, better than above, still misclustering and noise
f6 = plt.figure(6)
plot_clusters(data, cluster.AgglomerativeClustering, (),
              {'n_clusters': 5, 'linkage': 'ward'})

# DBSCAN - density based algorithm
# Performance: near optimal
f7 = plt.figure(7)
plot_clusters(data, cluster.DBSCAN, (), {'eps': 0.25})

# HDBSCAN - density based algorithm that allows for varying density
# Performance: optimal
# Add desaturation for points with lower probability of belonging
# to a cluster.
f8 = plt.figure(8)
clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0,
                            approx_min_span_tree=True,
                            gen_min_span_tree=True, leaf_size=40,
                            memory=Memory(cachedir=None),
                            metric='euclidean', min_cluster_size=6,
                            min_samples=None, p=None,
                            cluster_selection_method='eom')
clusterer = clusterer.fit(data)

start_time = time.time()
end_time = time.time()
palette = sns.color_palette('deep')
cluster_colors = [sns.desaturate(palette[col], sat)
                  if col >= 0 else (0.5, 0.5, 0.5) for col, sat in
                  zip(clusterer.labels_, clusterer.probabilities_)]
plt.scatter(data.T[0], data.T[1], c=cluster_colors, **plot_kwds)
frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)
plt.title('Clusters found by {} (eom)'.format(str(hdbscan.HDBSCAN.__name__)),
          fontsize=24)
plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time),
         fontsize=14)

# HDBSCAN as above, now with leaf cluster_selection_method, which necessitates
# adaptation of min_cluster_size
f9 = plt.figure(9)
clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0,
                            approx_min_span_tree=True,
                            gen_min_span_tree=True, leaf_size=40,
                            memory=Memory(cachedir=None),
                            metric='euclidean', min_cluster_size=15,
                            min_samples=None, p=None,
                            cluster_selection_method='leaf')
clusterer = clusterer.fit(data)

start_time = time.time()
end_time = time.time()
palette = sns.color_palette('deep')
cluster_colors = [sns.desaturate(palette[col], sat)
                  if col >= 0 else (0.5, 0.5, 0.5) for col, sat in
                  zip(clusterer.labels_, clusterer.probabilities_)]
plt.scatter(data.T[0], data.T[1], c=cluster_colors, **plot_kwds)
frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)
plt.title('Clusters found by {} (leaf)'.format(str(hdbscan.HDBSCAN.__name__)),
          fontsize=24)
plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time),
         fontsize=14)

plt.show()
