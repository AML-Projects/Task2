from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import AllKNN, ClusterCentroids, RandomUnderSampler, NearMiss


class DataSampling:
    def __init__(self, over_sampling_method):
        self.over_sampling_method = over_sampling_method
        print(self.over_sampling_method)

    def sampling(self, X, y):
        # oversampling
        if self.over_sampling_method == "random":
            return RandomOverSampler(random_state=41).fit_resample(X, y)

        if self.over_sampling_method == "SMOTE":
            return SMOTE(n_jobs=-1, random_state=41).fit_resample(X, y)

        if self.over_sampling_method == "ADASYN":
            return ADASYN(n_jobs=-1, random_state=41).fit_resample(X, y)

        # under sampling
        if self.over_sampling_method == "RandomUnderSampler":
            return RandomUnderSampler(random_state=0).fit_resample(X, y)

        if self.over_sampling_method == "ClusterCentroids":
            return ClusterCentroids(random_state=41).fit_resample(X, y)

        if self.over_sampling_method == "NearMiss":
            return NearMiss().fit_resample(X, y)

        if self.over_sampling_method == "AllKNN":
            return AllKNN(n_jobs=-1).fit_resample(X, y)

        # over and under sampling
        if self.over_sampling_method == "SMOTEENN":
            return SMOTEENN(n_jobs=-1, random_state=41).fit_resample(X, y)

        if self.over_sampling_method == "SMOTETomek":
            return SMOTETomek(n_jobs=-1, random_state=41).fit_resample(X, y)

    def fit_resample(self, X, y):
        if self.over_sampling_method == "None":
            return X, y
        return self.sampling(X, y)


def preprocess_data(x_train, y_train, x_val, y_val):
    ds = DataSampling("ADASYN")
    x_train, y_train = ds.fit_resample(x_train, y_train)

    return x_train, y_train, x_val, y_val
