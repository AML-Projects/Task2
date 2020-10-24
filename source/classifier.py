import math
from datetime import datetime

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier, RUSBoostClassifier, \
    BalancedBaggingClassifier
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import class_weight
from tensorflow.python.keras.optimizer_v2.nadam import Nadam

from logcreator.logcreator import Logcreator


def xgb_classifier(x_train, y_train, weights):
    model = xgb.XGBClassifier(objective='multi:softmax', max_depth=1, num_class=3, random_state=41)

    model.fit(x_train, y_train.values.ravel(), sample_weight=weights)

    return model


def baseline_model(nr_features):
    model = Sequential()
    model.add(Dense(128, input_dim=nr_features,
                    kernel_initializer='normal',
                    kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.005, l2=0.001),
                    activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128,
                    kernel_initializer='normal',
                    kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.003, l2=0.001),
                    activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64,
                    kernel_initializer='normal',
                    kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.001, l2=0.001),
                    activation='relu'))
    model.add(Dense(3, activation='softmax'))

    METRICS = [
        keras.metrics.CategoricalAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
        keras.metrics.CategoricalCrossentropy('crossentropy')
    ]

    opt = Nadam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=METRICS)

    # keras.utils.plot_model(model, to_file='./trainings/model.png', show_shapes=True, rankdir="LR")
    return model


def neural_network(x, y, class_weights, nr_features):
    model = baseline_model(nr_features)

    # model = KerasClassifier(build_fn=model)
    params = {"epochs": 200,
              "validation_split": 0.2,
              "class_weight": class_weights,
              "batch_size": 32,
              "verbose": 0
              }

    log_tensorboard = True
    if log_tensorboard:
        logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
        params["callbacks"] = [tensorboard_callback]
        # to look at the data start the tensorboard in the terminal
        # tensorboard --logdir logs/scalars

    model.fit(x, y, **params)

    return model


class Classifier:
    def __init__(self, classifier):
        self.classifier = classifier
        Logcreator.info(self.classifier)

    def fit(self, X, y):
        class_weights, class_weights_dict = self.compute_class_weights(y)

        params = {}

        if self.classifier == "BalancedRandomForestClassifier":
            model = BalancedRandomForestClassifier(n_estimators=100, random_state=0)

        elif self.classifier == "EasyEnsembleClassifier":
            model = EasyEnsembleClassifier(random_state=0)

        elif self.classifier == "BBC":
            model = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                              sampling_strategy='auto',
                                              replacement=False,
                                              random_state=0)

        elif self.classifier == "GaussianNB":
            model = GaussianNB()

        elif self.classifier == "RUSBoostClassifier":
            model = RUSBoostClassifier(n_estimators=200, algorithm='SAMME.R', random_state=0)

        elif self.classifier == "xgb":
            weights = y.copy()
            for i in range(0, len(class_weights)):
                weights.loc[weights.y == i] = class_weights[i]

            model = xgb_classifier(X, y, weights)
            return model

        elif self.classifier == "LogisticRegression":
            model = LogisticRegression(penalty='l2',
                                       C=1,
                                       class_weight=class_weights_dict,
                                       max_iter=100,
                                       n_jobs=-1,
                                       random_state=0)

        elif self.classifier == "SVC":
            model = SVC(C=1, class_weight=class_weights_dict, random_state=41)
            params['C'] = [1]
            params['kernel'] = ['rbf', 'linear', 'poly']

        elif self.classifier == "NN":
            # y to one hot encoding
            y = tf.keras.utils.to_categorical(y, 3)

            nr_features = X.shape[1]
            model = neural_network(X, y, class_weights_dict, nr_features)

            return model
        else:
            raise ValueError("Model not existing")

        nr_folds = math.floor(math.sqrt(X.shape[0]) / 2)

        best_model, results = self.do_grid_search(model, nr_folds, parameters=params, X=X, y=y.values.ravel())

        return best_model

    @staticmethod
    def do_grid_search(model, nr_folds, parameters, X, y):
        Logcreator.info("\nStarting Grid Search\n")

        skf = StratifiedKFold(shuffle=True, n_splits=nr_folds, random_state=41)

        grid_search = GridSearchCV(model, parameters,
                                   scoring='balanced_accuracy',
                                   # use every cpu thread
                                   n_jobs=-1,
                                   # Refit an estimator using the best found parameters
                                   refit=True,
                                   cv=skf,
                                   # Return train score to check for overfitting
                                   return_train_score=True,
                                   verbose=1)

        grid_search.fit(X, y)

        # Best estimator
        Logcreator.info("Best estimator from GridSearch: {}".format(grid_search.best_estimator_))
        Logcreator.info("Best alpha found: {}".format(grid_search.best_params_))
        Logcreator.info("Best training-score with mse loss: {}".format(grid_search.best_score_))

        results = pd.DataFrame(grid_search.cv_results_)
        results.sort_values(by='rank_test_score', inplace=True)

        Logcreator.info(
            results[['params', 'mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score']].head(30))

        best_model = grid_search.best_estimator_

        return best_model, results

    @staticmethod
    def compute_class_weights(y):
        classes = np.unique(y['y'])
        number_classes = len(classes)
        class_weights = list(class_weight.compute_class_weight(class_weight='balanced',
                                                               classes=classes,
                                                               y=y['y']))
        Logcreator.info("\nClass weights:\n", pd.DataFrame(class_weights))

        Logcreator.info("\nSamples per group before classification\n", y.groupby("y")["y"].count())

        return class_weights, dict(zip(range(len(class_weights)), class_weights))
