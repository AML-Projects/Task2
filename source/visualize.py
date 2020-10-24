import os

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from source.configuration import Configuration


def visualize_prediction(x, y_true, y_pred, title='', save_img=False, interactive=False):
    pca = PCA(n_components=3)

    components = pca.fit_transform(x)

    if interactive:
        # pip install PyQt5
        import matplotlib as mpl
        mpl.use("Qt5Agg")

    fig = plt.figure(figsize=(12, 12))
    fig.suptitle(title + ": True vs Predicted", fontsize=20)
    fig.set_dpi(100)

    # plot ture values
    subplot_components(components, 211, "True labels", y_true, elev=45.0)
    subplot_components(components, 212, "Predicted", y_pred, elev=45.0)

    plt.show()
    if save_img:
        plt.savefig(os.path.join(Configuration.output_directory, title + "_True_VS_Predicted" + ".png"))


def plot_true_labels(x, y, components, title="", save_img=True, interactive=False):
    if interactive:
        # pip install PyQt5
        import matplotlib as mpl
        mpl.use("Qt5Agg")

    fig = plt.figure(figsize=(12, 12))
    fig.suptitle(title, fontsize=20)
    fig.set_dpi(150)

    # plot ture values
    y_pred = y.values.T
    subplot_components(components, 111, "True labels", y_pred)
    if save_img:
        plt.savefig(fname="./img/extraction/" + title + ".png")


def subplot_components(components, plot_nr, plot_title, y_pred, elev=65.0, azim=45.0):
    if components.shape[1] == 3:
        ax = plt.subplot(plot_nr, projection='3d')
        ax.autoscale(tight=True)
        ax.scatter(components[:, 0], components[:, 1], components[:, 2], c=y_pred)
        ax.elev = elev
        ax.azim = azim
        ax.set_zlabel('component 2')
    else:
        plt.subplot(plot_nr)
        plt.scatter(components[:, 0], components[:, 1], c=y_pred)
    plt.xlabel('component 0')
    plt.ylabel('component 1')
    plt.title(plot_title)
