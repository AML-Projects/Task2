from sklearn.base import TransformerMixin
from sklearn.kernel_approximation import Nystroem


class FeatureSelector(TransformerMixin):
    def __init__(self, method="None"):
        self.method = method
        print("\nData Sampling:", self.method)
        self.feature_map_nystroem = Nystroem(
            gamma=None,
            n_components=200,
            random_state=41)

    def fit(self, X, y):
        if self.method == "Nystroem":
            return self.feature_map_nystroem.fit(X, y)
        else:
            return self

    def transform(self, X):
        if self.method == "Nystroem":
            return self.feature_map_nystroem.transform(X)
        else:
            return X
