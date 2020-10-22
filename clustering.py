import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def visualize(x, y):
    """
    https://scikit-learn.org/stable/modules/clustering.html

    :param x:
    :param y:
    :return:
    """

    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(x, y)

    """
    explained_variance = pca.explained_variance_ratio_

    print("pca explained variance", explained_variance)

    principal_components_T = principal_components.T
    for i in range(0, nr_components):
        fig, ax = plt.subplots()
        plt.scatter(principal_components_T[i], y.astype(int))
        ax.set(xlabel='pc' + str(i + 1), ylabel='class', title='principle component ' + str(i + 1) + ' vs target label')
        ax.grid()
        plt.show()
    """

    # lda returns max 2 components
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda_components = lda.fit_transform(x, y)

    # fa = FeatureAgglomeration(n_clusters=3, linkage='ward')
    # feature_agglo = fa.fit_transform(x, y)

    cluster_dict = {
        "KMeans": KMeans(n_clusters=3, random_state=41),
        "Birch": Birch(threshold=0.8, n_clusters=3),
        # linkage : {'ward', 'complete', 'average', 'single'}
        "AgglomerativeClustering": AgglomerativeClustering(n_clusters=3, linkage='ward'),
    }

    for name, clusterer in cluster_dict.items():
        plot_predicted_clusters(x, y, principal_components, clusterer, "PCA - " + name)
        plot_predicted_clusters(x, y, lda_components, clusterer, "LDA - " + name)
        # plot_predicted_clusters(x, y, feature_agglo, clusterer, "FeatureAgglomeration - " + name)
        plt.show()

    return principal_components


def plot_predicted_clusters(x, y, components, clusterer, title="", interactive=False):
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
    subplot_components(components, 223, "Predicted on only the supplied components", y_pred)


def subplot_components(components, plot_nr, plot_title, y_pred):
    if components.shape[1] == 3:
        ax = plt.subplot(plot_nr, projection='3d')
        ax.autoscale(tight=True)
        ax.scatter(components[:, 0], components[:, 1], components[:, 2], c=y_pred)
    else:
        plt.subplot(plot_nr)
        plt.scatter(components[:, 0], components[:, 1], c=y_pred)
    plt.xlabel('comp 0')
    plt.ylabel('comp 1')
    plt.title(plot_title)
