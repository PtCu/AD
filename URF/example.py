from sklearn.ensemble import RandomTreesEmbedding
from sklearn import cluster, manifold

"""
Random Forest clustering works as follows
1. Construct a dissimilarity measure using RF
2. Use an embedding algorithm (MDS, TSNE) to embed into a 2D space preserving that dissimilarity measure.
3. Cluster using K-means or K-medoids
"""
X = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]]
random_trees = RandomTreesEmbedding(
    n_estimators=5, random_state=0, max_depth=1).fit(X)

X_sparse_leaves = random_trees.fit_transform(X)
X_sparse_leaves.toarray()
projector = manifold.TSNE(random_state=1234, metric="euclidean")
X_sparse_embedding=projector.fit_transform(X_sparse_leaves)

clusterer = cluster.KMeans(n_clusters=2, random_state=1234, n_init=20)
clusterer.fit(X_sparse_embedding)
label=clusterer.labels_

