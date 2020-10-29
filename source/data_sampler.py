from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import AllKNN, ClusterCentroids, RandomUnderSampler, NearMiss

from logcreator.logcreator import Logcreator


class DataSampling:
    def __init__(self, sampling_method):
        if sampling_method == "None":
            sampling_method = None
        self.sampling_method = sampling_method
        Logcreator.info("\nData Sampling:", self.sampling_method)

    def sampling(self, X, y):
        # oversampling
        sampler = None
        if self.sampling_method == "random":
            sampler = RandomOverSampler(random_state=41)

        if self.sampling_method == "SMOTE":
            sampler = SMOTE(n_jobs=-1, random_state=41)

        if self.sampling_method == "ADASYN":
            sampler = ADASYN(n_jobs=-1, random_state=41)

        # under sampling
        if self.sampling_method == "RandomUnderSampler":
            sampler = RandomUnderSampler(random_state=0)

        if self.sampling_method == "ClusterCentroids":
            sampler = ClusterCentroids(random_state=41)

        if self.sampling_method == "NearMiss":
            sampler = NearMiss()  # .fit_resample(X, y)

        if self.sampling_method == "AllKNN":
            sampler = AllKNN(sampling_strategy='not minority', n_jobs=-1)

        # over and under sampling
        if self.sampling_method == "SMOTEENN":
            sampler = SMOTEENN(n_jobs=-1, random_state=41)

        if self.sampling_method == "SMOTETomek":
            sampler = SMOTETomek(n_jobs=-1, random_state=41)

        if sampler is None:
            Logcreator.warn("sampling method not found")
            return X, y

        x_sampled, y_sampled = sampler.fit_resample(X, y)

        Logcreator.info("\nResampled shape\n", x_sampled.shape)

        return x_sampled, y_sampled

    def fit_resample(self, X, y):
        if self.sampling_method is None:
            return X, y
        return self.sampling(X, y)
