from sklearn.base import TransformerMixin
from sklearn.manifold import Isomap

from logcreator.logcreator import Logcreator


class FeatureTransformer(TransformerMixin):
    def __init__(self, method="None", n_components=1000):
        self.method = method
        self.n_components = n_components
        self.transformer = None
        Logcreator.info("\nFeature Transformer:", self.method)

    def fit(self, X, y=None):
        if self.method == "Isomap":
            self.transformer = Isomap(n_components=self.n_components,
                                      n_jobs=-1)
        else:
            return self

        self.transformer.fit(X, y)

        return self

    def transform(self, X):
        if self.transformer is not None:
            return self.transformer.transform(X)
        else:
            return X
