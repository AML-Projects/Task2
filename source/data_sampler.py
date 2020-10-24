from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import AllKNN, ClusterCentroids, RandomUnderSampler, NearMiss


class DataSampling:
    def __init__(self, over_sampling_method):
        self.over_sampling_method = over_sampling_method
        print("\nData Sampling:", self.over_sampling_method)

    def sampling(self, X, y):
        # oversampling
        if self.over_sampling_method == "random":
            sampler = RandomOverSampler(random_state=41)

        if self.over_sampling_method == "SMOTE":
            sampler = SMOTE(n_jobs=-1, random_state=41)

        if self.over_sampling_method == "ADASYN":
            sampler = ADASYN(n_jobs=-1, random_state=41)

        # under sampling
        if self.over_sampling_method == "RandomUnderSampler":
            sampler = RandomUnderSampler(random_state=0)

        if self.over_sampling_method == "ClusterCentroids":
            sampler = ClusterCentroids(random_state=41)

        if self.over_sampling_method == "NearMiss":
            sampler = NearMiss()#.fit_resample(X, y)

        if self.over_sampling_method == "AllKNN":
            sampler = AllKNN(sampling_strategy='not minority', n_jobs=-1)

        # over and under sampling
        if self.over_sampling_method == "SMOTEENN":
            sampler = SMOTEENN(n_jobs=-1, random_state=41)

        if self.over_sampling_method == "SMOTETomek":
            sampler = SMOTETomek(n_jobs=-1, random_state=41)

        x_sampled, y_sampled = sampler.fit_resample(X, y)

        print("\nResampled shape\n", x_sampled.shape)

        return x_sampled, y_sampled

    def fit_resample(self, X, y):
        if self.over_sampling_method == "None":
            return X, y
        return self.sampling(X, y)


def preprocess_data(x_train, y_train, x_val, y_val):
    ds = DataSampling("ADASYN")
    x_train, y_train = ds.fit_resample(x_train, y_train)

    return x_train, y_train, x_val, y_val
