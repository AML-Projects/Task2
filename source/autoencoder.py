import numpy as np
import pandas
from keras.layers import Dense, Input
from keras.models import Model
from keras.regularizers import l1_l2
from numpy.random import seed
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.utils import class_weight
from tensorflow.python.framework.random_seed import set_random_seed
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.optimizer_v2.adamax import Adamax

from helpers.evaluation import evaluate_bas
from helpers.visualize import visualize_prediction
from logcreator.logcreator import Logcreator


class AutoEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, add_noise=False, encoded_size=64, scaling_on=True, use_sample_weight=True, verbose=1):
        self.add_noise = add_noise
        self.encoded_size = encoded_size
        self.scaling_on = scaling_on
        self.use_sample_weight = use_sample_weight
        self.verbose = verbose
        if self.verbose == 1:
            Logcreator.info("\nAuto encoder:")
            Logcreator.info("add_noise =", add_noise)
            Logcreator.info("encoded size =", encoded_size)
            Logcreator.info("scaling_on =", scaling_on)
        pass

    def scale_input(self, x):
        if self.scaling_on:
            scaler = MinMaxScaler(feature_range=(0, 1))
            x = scaler.fit_transform(x)

        return x

    def fit(self, X, y=None, *_):
        """
        https://towardsdatascience.com/applied-deep-learning-part-3-autoencoders-1c083af4d798
        https://blog.keras.io/building-autoencoders-in-keras.html

        :param X:
        :return:
        """
        input_size = X.shape[1]
        hidden_size = 256
        encoded_size = self.encoded_size

        X = self.scale_input(X)

        # add noise to input
        if self.add_noise:
            noise_factor = 0.3
            x_train_noisy = X + noise_factor * np.random.normal(size=X.shape)
            x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
            X = x_train_noisy

        input_img = Input(shape=(input_size,))
        hidden_1 = Dense(hidden_size, activation='elu', activity_regularizer=l1_l2(10e-6, 10e-6))(input_img)
        encoded = Dense(encoded_size, activation='elu', activity_regularizer=l1_l2(10e-6, 10e-6))(hidden_1)
        hidden_2 = Dense(hidden_size, activation='elu', activity_regularizer=l1_l2(10e-6, 10e-6))(encoded)
        decoded = Dense(input_size, activation='sigmoid')(hidden_2)

        autoencoder = Model(input_img, decoded)

        # This model maps an input to its encoded representation
        self.encoder = Model(input_img, encoded)

        optimizer = Adamax(learning_rate=0.0005)
        # optimizer = Nadam(learning_rate=0.0001)

        autoencoder.compile(optimizer=optimizer, loss='mean_squared_error',
                            metrics=['binary_crossentropy', 'mean_absolute_error'])

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=self.verbose, patience=8)

        if self.use_sample_weight:
            classes = np.unique(y['y'])
            class_weights = list(class_weight.compute_class_weight(class_weight='balanced',
                                                                   classes=classes,
                                                                   y=y['y']))
            weights = y.copy()
            for i in range(0, len(class_weights)):
                weights.loc[weights.y == i] = class_weights[i]

            history = autoencoder.fit(X, X, epochs=200, batch_size=10, callbacks=[es],
                                      sample_weight=weights,
                                      shuffle=True,
                                      validation_split=0.2,
                                      verbose=self.verbose)
        else:
            history = autoencoder.fit(X, X, epochs=200, batch_size=10, callbacks=[es],
                                      shuffle=True,
                                      validation_split=0.2,
                                      verbose=self.verbose)

        if self.verbose == 1:
            Logcreator.info("Auto encoder parameter:", history.params)
            Logcreator.info("Auto encoder fit results epoch 0 loss:", history.history['loss'][0], "; val_loss:",
                            history.history['val_loss'][0])
            final_epoch = (len(history.epoch) - 1)
            Logcreator.info("Auto encoder fit results epoch", final_epoch, "loss:",
                            history.history['loss'][final_epoch], "; val_loss:",
                            history.history['val_loss'][final_epoch])

        # evaluate performance
        evaluate = False
        if evaluate:
            clf = SVC(class_weight='balanced')
            clf.fit(self.encoder.predict(X), y)
            y_pred = clf.predict(self.encoder.predict(X))

            evaluate_bas(y_true=y, y_predicted=y_pred, text="Auto Encoder with default SVC on Train")

            visualize_prediction(X, y, y_pred, "Auto Encoded")

        exit()
        return self

    def transform(self, X, *_):
        X = self.scale_input(X)
        return self.encoder.predict(X)


def grid_search_auto_encoder():
    x_train = pandas.read_csv("../data/X_train.csv", index_col=0)
    y_train = pandas.read_csv("../data/y_train.csv", index_col=0)

    AutoEncoder().fit(x_train, y_train)

    pipeline = Pipeline([('scaler', StandardScaler()),
                         ('ae', AutoEncoder(verbose=0)),
                         ('svc', SVC())])

    parameters = {
        'ae__add_noise': [False, True],
        'ae__scaling_on': [False, True],
        'ae__use_sample_weight': [False, True],
        'ae__encoded_size': [16, 32, 64, 128],
    }

    skf = StratifiedKFold(shuffle=True, n_splits=4, random_state=41)
    searcher = GridSearchCV(pipeline, parameters,
                            scoring='balanced_accuracy',
                            # use every cpu thread
                            n_jobs=-3,
                            # Refit an estimator using the best found parameters
                            refit=True,
                            cv=skf,
                            # Return train score to check for overfitting
                            return_train_score=True,
                            verbose=1)

    searcher.fit(x_train, y_train)

    Logcreator.info("Best estimator from GridSearch: {}".format(searcher.best_estimator_))
    Logcreator.info("Best alpha found: {}".format(searcher.best_params_))
    Logcreator.info("Best training-score with mse loss: {}".format(searcher.best_score_))
    results = pandas.DataFrame(searcher.cv_results_)
    results.sort_values(by='rank_test_score', inplace=True)
    Logcreator.info(
        results[['params', 'mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score']].head(30))

    pandas.DataFrame.to_csv(results, '../trainings/ae_search_results.csv')


if __name__ == "__main__":
    seed(1)
    set_random_seed(1)

    pandas.set_option('display.max_rows', None)
    pandas.set_option('display.max_columns', None)
    pandas.set_option('display.width', None)
    pandas.set_option('display.max_colwidth', None)

    # check_estimator(AutoEncoder)
    grid_search_auto_encoder()
