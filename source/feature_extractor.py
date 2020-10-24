from sklearn.base import TransformerMixin
from sklearn.cross_decomposition import CCA, PLSRegression
from sklearn.decomposition import FastICA
from sklearn.kernel_approximation import Nystroem
from sklearn.manifold import LocallyLinearEmbedding, TSNE

from logcreator.logcreator import Logcreator


class FeatureExtractor(TransformerMixin):
    def __init__(self, method="None", n_components=200):
        self.method = method
        self.n_components = n_components
        self.transformer = None
        Logcreator.info("\nFeature Extraction:", self.method)

    def fit(self, X, y):
        if self.method == "Nystroem":
            self.transformer = Nystroem(
                gamma=None,
                n_components=self.n_components,
                random_state=41)

        elif self.method == "FastICA":
            self.transformer = FastICA(n_components=self.n_components, random_state=41)

        elif self.method == "LLE":
            self.transformer = LocallyLinearEmbedding(n_components=self.n_components, random_state=41)

        elif self.method == "TSNE":
            self.transformer = TSNE(n_components=self.n_components, random_state=41)

        elif self.method == "CCA":
            self.transformer = CCA(n_components=2)

        elif self.method == "PLSRegression":
            self.transformer = PLSRegression(n_components=self.n_components)

        else:
            return self

        self.transformer.fit(X, y)
        return self

    def transform(self, X):
        if self.transformer is not None:
            return self.transformer.transform(X)
        else:
            return X
