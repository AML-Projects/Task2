import os

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

from helpers.visualize import visualize_true_labels
from logcreator.logcreator import Logcreator


class DataSplitter:
    def __init__(self):
        pass

    def split_data(self, x, y=None, log_text="", boundary=0):
        """
        Split data according to PCA component 0 according to the supplied boundary.
        :param x:
        :param y:
        :param boundary:
        :return:
        """
        pca = PCA(n_components=1)

        # scale x
        scaler = RobustScaler()
        x_scaled = scaler.fit_transform(x)

        visualize_true_labels(x, y_train, "Before split - " + log_text)

        components = pca.fit_transform(x_scaled)
        components = pd.DataFrame(components)

        remove_rows = (components > boundary)

        self.remove_idx1 = remove_rows.index[remove_rows[0] == False].tolist()
        self.remove_idx2 = remove_rows.index[remove_rows[0] == True].tolist()

        x1 = x.drop(index=self.remove_idx2)
        x2 = x.drop(index=self.remove_idx1)

        if y is not None:
            y1 = y.drop(index=self.remove_idx2)
            y2 = y.drop(index=self.remove_idx1)

            visualize_true_labels(x1, y1, title="Split 1 - " + log_text)
            visualize_true_labels(x2, y2, title="Split 2 - " + log_text)

            return x1, y1, x2, y2

        visualize_true_labels(x1, y_true=None, title="Split 1 - " + log_text)
        visualize_true_labels(x2, y_true=None, title="Split 2 - " + log_text)

        return x1, x2

    def combine_split(self, x1, x2):
        """
        Combine according to remove index
        :param x1:
        :param x2:
        :return:
        """
        if isinstance(x1, pd.DataFrame):
            x1 = x1.values
        if isinstance(x2, pd.DataFrame):
            x1 = x2.values
        x1 = pd.DataFrame(x1, index=self.remove_idx1)
        x2 = pd.DataFrame(x2, index=self.remove_idx2)

        return pd.concat([x1, x2]).sort_index()


def save_to_file(data, f_name):
    pd.DataFrame.to_csv(data, os.path.join("../data/split/", f_name), index=True)


if __name__ == '__main__':
    x_train = pd.read_csv("../data/X_train.csv", index_col=0)
    y_train = pd.read_csv("../data/y_train.csv", index_col=0)
    x_test = pd.read_csv("../data/X_test.csv", index_col=0)

    # before split
    Logcreator.info("\nBefore split - Train\n", y_train.groupby("y")["y"].count().values)

    # split data
    boundary = 5

    splitter_train = DataSplitter()
    x1_train, y1_train, x2_train, y2_train = splitter_train.split_data(x_train, y_train,
                                                                       log_text="Train",
                                                                       boundary=boundary)

    splitter_test = DataSplitter()
    x1_test, x2_test = splitter_test.split_data(x_test, log_text="Test", boundary=boundary)

    # after split
    Logcreator.info("\nSplit 1 - Train\n", y1_train.groupby("y")["y"].count().values)
    Logcreator.info("\nSplit 2 - Train\n", y2_train.groupby("y")["y"].count().values)

    # save to file
    save_to_file(x1_train, 'x1_train.csv')
    save_to_file(y1_train, 'y1_train.csv')
    save_to_file(x2_train, 'x2_train.csv')
    save_to_file(y2_train, 'y2_train.csv')

    save_to_file(x1_test, 'x1_test.csv')
    save_to_file(x2_test, 'x2_test.csv')
