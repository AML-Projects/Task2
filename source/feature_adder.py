import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from logcreator.logcreator import Logcreator
from source.autoencoder import AutoEncoder


class ClusterFeatureGenerator(BaseEstimator):
    def __init__(self, n_clusters=16):
        self.n_clusters = n_clusters
        self.clusterer = None
        pass

    def fit(self, x, y=None):
        return self

    def predict(self, x):
        # AgglomerativeClustering can only do fit_predict in one step
        self.clusterer = AgglomerativeClustering(n_clusters=self.n_clusters)
        y_pred = self.clusterer.fit_predict(x)

        return y_pred


class CustomFeatureGenerator:
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def predict(self, x):
        extracted_features = []

        # number of columns to extract the features from
        step_size = 1000
        column_range = range(0, x.shape[1], step_size)

        for index in column_range:
            current_columns = x[:, index:index + step_size]

            self.extract_values_from_columns(current_columns, extracted_features)

        return np.array(extracted_features).T

    def extract_values_from_columns(self, current_columns, extracted_features):
        # a_min = numpy.amin(current_columns, axis=1)
        a_max = np.amax(current_columns, axis=1)

        # max - min
        ptp = np.ptp(current_columns, axis=1)

        # percentile25 = numpy.percentile(current_columns, q=25, axis=1)
        percentile50 = np.percentile(current_columns, q=50, axis=1)
        percentile75 = np.percentile(current_columns, q=75, axis=1)

        # median is 50 percentil
        # median = numpy.median(current_columns, axis=1)

        # average = numpy.average(current_columns, axis=1)

        # mean = numpy.mean(current_columns, axis=1)

        std = np.std(current_columns, axis=1)

        # var = numpy.var(current_columns, axis=1)

        # gradient = numpy.max(numpy.gradient(current_columns, axis=1), axis=1)
        # trapeze = numpy.trapz(current_columns, axis=1)
        # diff = numpy.max(numpy.diff(current_columns, axis=1), axis=1)

        # extracted_features.append(a_min)
        extracted_features.append(a_max)
        extracted_features.append(ptp)
        # extracted_features.append(percentile25)
        extracted_features.append(percentile50)
        extracted_features.append(percentile75)

        # extracted_features.append(median)
        # extracted_features.append(average)
        # extracted_features.append(mean)
        extracted_features.append(std)
        # extracted_features.append(var)

        # extracted_features.append(gradient)
        # extracted_features.append(trapeze)
        # extracted_features.append(diff)

        return extracted_features


class FeatureAdder(TransformerMixin):
    def __init__(self, clustering_on=False, n_clusters=16,
                 custom_on=False,
                 auto_encoder_on=False, n_encoder_features=16):
        Logcreator.info("\nFeature Adder:")

        if isinstance(clustering_on, str):
            clustering_on = clustering_on == "True"
        self.clustering_on = clustering_on
        self.n_clusters = n_clusters
        if self.clustering_on:
            Logcreator.info("[clustering_on]")
            self.clusterFG = ClusterFeatureGenerator()

        if isinstance(custom_on, str):
            custom_on = custom_on == "True"
        self.custom_on = custom_on
        if self.custom_on:
            Logcreator.info("[custom_on]")
            self.customFG = CustomFeatureGenerator()

        if isinstance(auto_encoder_on, str):
            auto_encoder_on = auto_encoder_on == "True"
        self.auto_encoder_on = auto_encoder_on
        self.n_encoder_features = n_encoder_features
        if self.auto_encoder_on:
            Logcreator.info("[auto_encoder_on]")
            self.ae = AutoEncoder(encoded_size=self.n_encoder_features, scaling_on=True, add_noise=True)

        pass

    def fit(self, x, y=None):
        if self.clustering_on:
            self.clusterFG.fit(x)

        if self.custom_on:
            self.customFG.fit(x)

        if self.auto_encoder_on:
            self.ae.fit(x, None)

        return self

    def transform(self, X):
        x_add = np.empty(shape=(X.shape[0], 0))
        if self.clustering_on:
            new_feature_cluster = self.clusterFG.predict(X)
            new_feature_cluster = new_feature_cluster[np.newaxis].T

            x_add = np.c_[new_feature_cluster, x_add]

        if self.custom_on:
            new_features_custom = self.customFG.predict(X)

            x_add = np.c_[new_features_custom, x_add]

        if self.auto_encoder_on:
            new_features_ae = self.ae.transform(X)

            x_add = np.c_[new_features_ae, x_add]

        if x_add.size > 0:
            # scale new features
            scaler = StandardScaler()
            x_add = scaler.fit_transform(x_add)

            # add features to X
            X = np.c_[X, x_add]

        return X
