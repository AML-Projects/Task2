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
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import class_weight
from tensorflow.python.keras.optimizer_v2.nadam import Nadam

from logcreator.logcreator import Logcreator


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
    def __init__(self, classifier, random_search=True):
        self.classifier = classifier
        self.results = {}
        self.random_search = random_search
        Logcreator.info(self.classifier)

    def getModelAndParams(self, y):
        class_weights, class_weights_dict = self.compute_class_weights(y)

        params = {}
        fit_params = {}

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
            model = xgb.XGBClassifier(objective='multi:softmax', max_depth=1, num_class=3, random_state=41)

            params['objective'] = ['multi:softmax']
            # setting max_depth to high results in overfitting
            params['max_depth'] = [3, 4, 5]
            # subsampling of rows: lower values of subsample can prevent overfitting
            params['subsample'] = [i / 10. for i in range(7, 11)]

            extensive = False
            if extensive:
                params['learning_rate'] = [0.1, 0.2, 0.3]
                params['n_estimators'] = [50, 100, 150]
                params['gamma'] = [0, 0.1, 0.2]
                params['min_child_weight'] = [0, 0.5, 1]
                params['max_delta_step'] = [0]
                params['colsample_bytree'] = [0.6, 0.8, 1]
                params['colsample_bylevel'] = [1]
                params['reg_alpha'] = [0, 1e-2, 1, 1e1]
                params['reg_lambda'] = [0, 1e-2, 1, 1e1]

            weights = y.copy()
            for i in range(0, len(class_weights)):
                weights.loc[weights.y == i] = class_weights[i]

            fit_params['sample_weight'] = weights

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

        else:
            raise ValueError("Model not existing")

        return model, params, fit_params

    def fit(self, X, y):
        class_weights, class_weights_dict = self.compute_class_weights(y)

        if self.classifier == "NN":
            # y to one hot encoding
            y = tf.keras.utils.to_categorical(y, 3)

            nr_features = X.shape[1]
            model = neural_network(X, y, class_weights_dict, nr_features)

            return model, {}
        else:
            model, model_param, fit_param = self.getModelAndParams(y)

        nr_folds = math.floor(math.sqrt(X.shape[0]) / 3)

        best_model, self.results = self.do_grid_search(model, nr_folds, parameters=model_param, fit_parameter=fit_param,
                                                       X=X,
                                                       y=y.values.ravel())

        return best_model

    def getFitResults(self):
        return self.results

    def do_grid_search(self, model, nr_folds, parameters, fit_parameter, X, y):
        Logcreator.info("\nStarting Parameter Search\n")

        searcher = Classifier.get_search_instance(model, nr_folds, parameters, self.random_search)
        searcher.fit(X, y, **fit_parameter)

        results = Classifier.evaluate_search_results(searcher)
        best_model = searcher.best_estimator_

        return best_model, results

    @staticmethod
    def evaluate_search_results(seracher):
        # Best estimator
        Logcreator.info("Best estimator from GridSearch: {}".format(seracher.best_estimator_))
        Logcreator.info("Best alpha found: {}".format(seracher.best_params_))
        Logcreator.info("Best training-score with mse loss: {}".format(seracher.best_score_))
        results = pd.DataFrame(seracher.cv_results_)
        results.sort_values(by='rank_test_score', inplace=True)
        Logcreator.info(
            results[['params', 'mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score']].head(30))
        return results

    @staticmethod
    def get_search_instance(model, nr_folds, parameters, random_search=False):
        skf = StratifiedKFold(shuffle=True, n_splits=nr_folds, random_state=41)

        if random_search:
            number_of_searches = 100
            searcher = RandomizedSearchCV(model, parameters,
                                          scoring='balanced_accuracy',
                                          n_iter=number_of_searches,
                                          # use every cpu thread
                                          n_jobs=-1,
                                          # Refit an estimator using the best found parameters
                                          refit=True,
                                          cv=skf,
                                          # Return train score to check for overfitting
                                          return_train_score=True,
                                          verbose=1)
        else:
            searcher = GridSearchCV(model, parameters,
                                    scoring='balanced_accuracy',
                                    # use every cpu thread
                                    n_jobs=-1,
                                    # Refit an estimator using the best found parameters
                                    refit=True,
                                    cv=skf,
                                    # Return train score to check for overfitting
                                    return_train_score=True,
                                    verbose=1)
        return searcher

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
