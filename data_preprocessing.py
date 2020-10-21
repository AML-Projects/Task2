from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN


class DataSampling:
    def __init__(self, over_sampling_method):
        self.over_sampling_method = over_sampling_method

    def sampling(self, X, y):
        # oversampling
        if self.over_sampling_method == "random":
            return RandomOverSampler(n_jobs=-1, random_state=41).fit_resample(X, y)

        if self.over_sampling_method == "SMOTE":
            return SMOTE().fit_resample(X, y)

        if self.over_sampling_method == "ADASYN":
            return ADASYN(n_jobs=-1, random_state=41).fit_resample(X, y)

        # over and under sampling
        if self.over_sampling_method == "SMOTEENN":
            return SMOTEENN(n_jobs=-1, random_state=41).fit_resample(X, y)

        if self.over_sampling_method == "SMOTETomek":
            return SMOTETomek(n_jobs=-1, random_state=41).fit_resample(X, y)

    def fit_resample(self, X, y):
        return self.sampling(X, y)


def preprocess_data(x_train, y_train, x_val, y_val):
    ds = DataSampling("SMOTEENN")
    x_train, y_train = ds.fit_resample(x_train, y_train)

    return x_train, y_train, x_val, y_val
