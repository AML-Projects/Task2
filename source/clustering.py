import matplotlib.pyplot as plt
import tensorflow
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, FeatureAgglomeration
from sklearn.cross_decomposition import PLSCanonical, CCA, PLSSVD, PLSRegression
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import LocallyLinearEmbedding, TSNE, MDS, Isomap
from sklearn.preprocessing import StandardScaler

from source.visualize import plot_true_labels, subplot_components


def visualize(x, y):
    """
    https://scikit-learn.org/stable/modules/clustering.html

    :param x:
    :param y:
    :return:
    """

    # make sure data is scaled
    scaler = StandardScaler()
    x = scaler.fit_transform(x, y)

    # cross decomposition
    cross_decomposers = {
        "PLSCanonical": PLSCanonical(n_components=2),  # 2 because the 3rd component is just zero
        "CCA": CCA(n_components=2),  # 2 because the 3rd component is just zero
        "PLSSVD": PLSSVD(n_components=3),
        "PLSRegression": PLSRegression(n_components=3),
    }

    for extractor_name, extractor in cross_decomposers.items():
        x_components, y_component = extractor.fit_transform(x, tensorflow.keras.utils.to_categorical(y, 3))
        plot_true_labels(x, y, x_components, extractor_name)

    feature_extractors = {
        # Discriminant Analysis
        # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.discriminant_analysis
        "LDA": LinearDiscriminantAnalysis(n_components=2),
        # decomposition
        # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition
        "PCA": PCA(n_components=3),
        "FastICA": FastICA(n_components=3, random_state=41),
        # Manifold Learning
        # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.manifold
        "Isomap": Isomap(n_components=3, n_jobs=-1),
        "LocallyLinearEmbedding": LocallyLinearEmbedding(n_components=3, n_jobs=-1, random_state=41),
        "MDS": MDS(n_components=3, n_jobs=-1, random_state=41),
        "TSNE": TSNE(n_components=3, n_jobs=-1, random_state=41),
        # Clustering
        # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster
        "FeatureAgglomeration": FeatureAgglomeration(n_clusters=3, linkage='ward')
    }

    cluster_dict = {
        "KMeans": KMeans(n_clusters=3, random_state=41),
        "Birch": Birch(threshold=0.8, n_clusters=3),
        # linkage : {'ward', 'complete', 'average', 'single'}
        "AgglomerativeClustering": AgglomerativeClustering(n_clusters=3, linkage='ward'),
    }
    for extractor_name, extractor in feature_extractors.items():
        components = extractor.fit_transform(x, y)
        plot_true_labels(x, y, components, extractor_name)

    plt.show()

    """
    for name, clusterer in cluster_dict.items():
        plot_predicted_clusters(x, y, components, clusterer, extractor_name + " - " + name)
    """


def plot_predicted_clusters(x, y, components, clusterer, title="", interactive=True):
    if interactive:
        # pip install PyQt5
        import matplotlib as mpl
        mpl.use("Qt5Agg")

    fig = plt.figure(figsize=(12, 12))
    fig.suptitle(title, fontsize=20)
    fig.set_dpi(150)

    # plot ture values
    y_pred = y.values.T
    subplot_components(components, 221, "True labels", y_pred)

    # plot cluster prediction based on x
    y_pred = clusterer.fit_predict(x)

    subplot_components(components, 222, "Predicted on whole input data", y_pred)

    # plot cluster prediction based only on the supplied components
    y_pred = clusterer.fit_predict(components)
    subplot_components(components, 212, "Predicted on only the supplied components", y_pred)


