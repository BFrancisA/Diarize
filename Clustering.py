from scipy.cluster.vq import vq, kmeans, whiten
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA



"""
Run simple Kmeans clustering to get maxClusters centroids.
Returns the centroids found.
"""
def runKMeansClustering(features, maxClusters):
    whitenedFeature = whiten(features)
    result = kmeans(whitenedFeature, maxClusters)
    centroids = result[0]
    distortion = result[1]
    return centroids, distortion



"""
Fit a GMM to the features.  Determine the number of speakers be finding the
lowest number of components that have the lowest silouette distortion measure.
Returns the predicted class number 0, 1,... for each feature.
Expect maxClusters >= 2.
"""
def runGaussianMixtureClustering(features, maxClusters):

    # Start with result with 2 clusters (aka components)
    clusters = 2
    clusterer = GaussianMixture(n_components=clusters)
    clusterer.fit(features)

    # Predict the cluster for each feature
    lastPredictedClusters = clusterer.predict(features)
    lastScore = silhouette_score(features, lastPredictedClusters)

    for componentCount in range(3, maxClusters):
        clusterer = GaussianMixture(n_components=componentCount)
        clusterer.fit(features)

        # Predict the cluster for each feature
        predictedClusters = clusterer.predict(features)
        score = silhouette_score(features, predictedClusters)

        if score >= lastScore:
            break
        else:
            lastPredictedClusters = predictedClusters
            lastScore = score
            clusters = componentCount


    return lastPredictedClusters, lastScore, clusters



"""
Fit a GMM to the features.  Determine the number of speakers be finding the
lowest number of components that have the lowest silouette distortion measure.
Returns the predicted class number 0, 1,... for each feature.
Expect maxClusters >= 2.

This version applies PCA to the data.
"""
def runGaussianMixtureClusteringPca(features, maxClusters):

    # Start with result with 2 clusters (aka components)
    clusters = 2

    pca = PCA(n_components=maxClusters)
    pca.fit(features)
    pca_features = pca.transform(features)  # Transform the features into reduced space.

    clusterer = GaussianMixture(n_components=clusters)
    clusterer.fit(pca_features)

    # Predict the cluster for each feature
    lastPredictedClusters = clusterer.predict(pca_features)
    lastScore = silhouette_score(pca_features, lastPredictedClusters)

    for componentCount in range(3, maxClusters):

        clusterer = GaussianMixture(n_components=componentCount)
        clusterer.fit(pca_features)

        # Predict the cluster for each feature
        predictedClusters = clusterer.predict(pca_features)
        score = silhouette_score(pca_features, predictedClusters)

        if score >= lastScore:
            break
        else:
            lastPredictedClusters = predictedClusters
            lastScore = score
            clusters = componentCount

    return lastPredictedClusters, lastScore, clusters






