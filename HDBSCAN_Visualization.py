###
# Visual breakdown of HDBSCAN clustering
# Adapted from https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html
###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import sklearn.datasets as data
import time
import hdbscan
from joblib import Memory

sns.set_context('poster')
sns.set_style('white')
sns.set_color_codes()
plot_kwds = {'alpha': 0.5, 's': 80, 'linewidths': 0}


# Make some data
moons, _ = data.make_moons(n_samples=75, noise=0.05)
blobs, _ = data.make_blobs(n_samples=100, centers=[(1, 2.25), (-1.0, 1.9)], cluster_std=0.2)
test_data = np.vstack([moons, blobs])

# Scatterplot some data
f1 = plt.figure(1)
plt.scatter(test_data.T[0], test_data.T[1], c='b', **plot_kwds)


# Create clustering
clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
                gen_min_span_tree=True, leaf_size=40, memory=Memory(cachedir=None),
                metric='euclidean', min_cluster_size=5, min_samples=5, p=None)
clusterer.fit(test_data)


# Steps of HDBSCAN:

# Space is transformed to find mutual reachability metric.
# Then, islands of dense data are identified.
# A minimum spanning tree (Prim's algorithm) is used, k-value of 5:
f2 = plt.figure(2)
clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)

# Cluster hierarchy is built:
f3 = plt.figure(3)
clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis',
                                      edge_alpha=0.6,
                                      node_size=80,
                                      edge_linewidth=2)

# Cluster tree is condensed using a minimum cluster size (here:5)
f4 = plt.figure(4)
clusterer.condensed_tree_.plot()

# Persistent long-lived clusters are extracted (greatest area of ink)
# This is the exess of mass method, alternative break by leaves;
# In HDBSCAN parameter cluster_selection_method = eom OR leaf :
f5 = plt.figure(5)
clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())

# Cluster labeling; color encodes cluster, color saturation encodes strength membership
# Unclustered points are gray
f6 = plt.figure(6)
palette = sns.color_palette()
palette2 = cm.rainbow(np.linspace(0,1,100)) # Larger range of colors for many clusters
cluster_colors = [sns.desaturate(palette[col], sat)
                  if col >= 0 else (0.5, 0.5, 0.5) for col, sat in
                  zip(clusterer.labels_, clusterer.probabilities_)]
plt.scatter(test_data.T[0], test_data.T[1], c=cluster_colors, **plot_kwds)

# Add an outlier analysis
f7 = plt.figure(7)
clusterer.outlier_scores_
sns.distplot(clusterer.outlier_scores_[np.isfinite(clusterer.outlier_scores_)],
             rug=True)

f8 = plt.figure(8)
threshold = pd.Series(clusterer.outlier_scores_).quantile(0.9)
outliers = np.where(clusterer.outlier_scores_ > threshold)[0]
plt.scatter(*test_data.T, s=50, linewidth=0, c='gray', alpha=0.25)
plt.scatter(*test_data[outliers].T, s=50, linewidth=0, c='red', alpha=0.5)

plt.show()
