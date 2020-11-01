import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import AgglomerativeClustering
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
        a_min = np.amin(current_columns, axis=1)
        extracted_features.append(a_min)

        a_max = np.amax(current_columns, axis=1)
        extracted_features.append(a_max)

        # max - min
        ptp = np.ptp(current_columns, axis=1)
        extracted_features.append(ptp)

        percentile25 = np.percentile(current_columns, q=25, axis=1)
        extracted_features.append(percentile25)

        percentile50 = np.percentile(current_columns, q=50, axis=1)
        extracted_features.append(percentile50)

        percentile75 = np.percentile(current_columns, q=75, axis=1)
        extracted_features.append(percentile75)

        # median is 50 percentile
        # median = np.median(current_columns, axis=1)
        # extracted_features.append(median)

        average = np.average(current_columns, axis=1)
        extracted_features.append(average)

        mean = np.mean(current_columns, axis=1)
        extracted_features.append(mean)

        std = np.std(current_columns, axis=1)
        extracted_features.append(std)

        var = np.var(current_columns, axis=1)
        extracted_features.append(var)

        gradient = np.max(np.gradient(current_columns, axis=1), axis=1)
        extracted_features.append(gradient)

        trapeze = np.trapz(current_columns, axis=1)
        extracted_features.append(trapeze)

        diff = np.max(np.diff(current_columns, axis=1), axis=1)
        extracted_features.append(diff)

        return extracted_features


class FeatureAdder(TransformerMixin):
    def __init__(self, clustering_on=False, n_clusters=16,
                 custom_on=False,
                 auto_encoder_on=False, n_encoder_features=16, encoder_path="",
                 lda_on=False,
                 lda_shrinkage=None,
                 replace_features=False):
        """
        :param clustering_on:
        :param n_clusters:
        :param custom_on:
        :param auto_encoder_on:
        :param n_encoder_features:
        :param encoder_path: load a pretrained encoder from the supplied path
        :param lda_on:
        :param lda_shrinkage:
        :param replace_features: True = Only keep extracted features; False = add extracted features to existing features
        """
        Logcreator.info("\nFeature Adder:")

        if isinstance(clustering_on, str):
            clustering_on = clustering_on == "True"
        self.clustering_on = clustering_on
        self.n_clusters = n_clusters
        if self.clustering_on:
            Logcreator.info("[clustering_on], n_clusters:", self.n_clusters)
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
        self.encoder_path = encoder_path
        if self.auto_encoder_on:
            Logcreator.info("[auto_encoder_on], n_encoder_features:", self.n_encoder_features)
            self.ae = AutoEncoder(encoded_size=self.n_encoder_features, scaling_on=True, add_noise=False,
                                  load_model_path=self.encoder_path)

        if isinstance(lda_on, str):
            lda_on = lda_on == "True"
        self.lda_on = lda_on
        if isinstance(lda_shrinkage, str):
            if lda_shrinkage == "None":
                lda_shrinkage = None
            elif lda_shrinkage != "auto":
                ValueError("lda_shrinkage can't have value", lda_shrinkage)
        if self.lda_on:
            Logcreator.info("[lda_on], shrinkage:", lda_shrinkage)
            """
            LDA can add  a lot of bias on the data, because it uses y in the fit!
            The shrinking parameter can regularize overfitting!
            So if the cross validation score mean_test_score on the training data goes up
            but on the test split does go down, we are likely overfitting.
            https://scikit-learn.org/stable/modules/lda_qda.html#shrinkage
            """
            self.lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage=lda_shrinkage)

        if isinstance(replace_features, str):
            replace_features = replace_features == "True"
        self.replace_features = replace_features

        pass

    def fit(self, x, y=None):
        if self.clustering_on:
            self.clusterFG.fit(x)

        if self.custom_on:
            self.customFG.fit(x)

        if self.auto_encoder_on:
            self.ae.fit(x, y)

        if self.lda_on:
            self.lda.fit(x, y)

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

        if self.lda_on:
            new_features_lda = self.lda.transform(X)

            x_add = np.c_[new_features_lda, x_add]

        if x_add.size > 0:
            # scale new features
            scaler = StandardScaler()
            x_add = scaler.fit_transform(x_add)

            if self.replace_features:
                # overwrite features
                X = x_add
            else:
                # add features to X
                X = np.c_[X, x_add]

        Logcreator.info("X shape after feature addition", X.shape)

        return X
