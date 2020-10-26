from collections import Counter

import numpy as np
import pandas as pd
from sklearn.utils import class_weight
import os
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import balanced_accuracy_score, multilabel_confusion_matrix, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import AllKNN, ClusterCentroids, RandomUnderSampler, NearMiss
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from source.visualize import visualize_prediction
from source import evaluation

from source.classifier import Classifier
from source.data_sampler import DataSampling


if __name__ == '__main__':
    # --------------------------------------------------------------------------------------------------------------
    # Read input data
    train_data_x = pd.read_csv("./data/X_train.csv")
    train_data_y = pd.read_csv("./data/y_train.csv")

    test_data = pd.read_csv("./data/X_test.csv")
    x_test = test_data.drop("id", axis=1)
    id_test = test_data["id"]

    del train_data_x['id']
    del train_data_y["id"]

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    print("Train-Data Shpae: " + str(train_data_x.shape))
    print("Test-Data Shape: " + str(x_test.shape))

    # --------------------------------------------------------------------------------------------------------------
    # Split
    # x_train, x_test, y_train, y_test = \
    #     model_selection.train_test_split(train_data_x, train_data_y,
    #                                      test_size=0.2,
    #                                      stratify=train_data_y,
    #                                      random_state=41)

    #print("\nTrain samples per group\n", train_data_x.groupby("y")["y"].count().values)
    print("\nValidation samples per group\n", train_data_y.groupby("y")["y"].count().values)

    # reset all indexes
    train_data_x.reset_index(drop=True, inplace=True)
    train_data_y.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)

    pca = PCA(n_components=100)
    components = pca.fit_transform(train_data_x)

    for i in range(1,10):

        fig = plt.figure(figsize=(12, 12))
        fig.suptitle("PCA of Training data", fontsize=20)
        fig.set_dpi(100)
        classes = ['c0', 'c1', 'c2']
        plt.scatter(components[:, 0], components[:, i], c=train_data_y.to_numpy())
        cb = plt.colorbar()
        # loc = np.arange(0, max(train_data_y), max(train_data_y) / float(len(classes)))
        cb.set_ticks([0, 1, 2])
        cb.set_ticklabels(classes)
        plt.xlabel('component 0')

        plt.ylabel('component ' + str(i))
        plt.show()

    # --------------------------------------------------------------------------------------------------------------
    # Split
    train_data_x, x_test_split, train_data_y, y_test_split = \
        model_selection.train_test_split(train_data_x, train_data_y, test_size=0.2, stratify=train_data_y, random_state=41)





    # --------------------------------------------------------------------------------------------------------------
    # # Sampling
    #train_data_x, train_data_y = SMOTEENN(n_jobs=-1, random_state=41).fit_resample(train_data_x, train_data_y)
    #train_data_x, train_data_y = RandomOverSampler(random_state=41).fit_resample(train_data_x, train_data_y)
    #train_data_x, train_data_y = SMOTE(n_jobs=-1, random_state=41).fit_resample(train_data_x, train_data_y)
    #train_data_x, train_data_y = ADASYN(n_jobs=-1, random_state=41).fit_resample(train_data_x, train_data_y)
    #train_data_x, train_data_y = RandomUnderSampler(random_state=0).fit_resample(train_data_x, train_data_y)
    train_data_x, train_data_y = ClusterCentroids(random_state=41).fit_resample(train_data_x, train_data_y)
    train_data_x, train_data_y = AllKNN(sampling_strategy='not minority', n_jobs=-1).fit_resample(train_data_x, train_data_y)
    train_data_x, train_data_y = SMOTETomek(n_jobs=-1, random_state=41).fit_resample(train_data_x, train_data_y)
    print("\n Trainset samples per group\n", train_data_y.groupby("y")["y"].count().values)
    print("\n Testset samples per group\n", y_test_split.groupby("y")["y"].count().values)
    # --------------------------------------------------------------------------------------------------------------
    # Scaling
    scaler = RobustScaler()
    #
    train_data_x = scaler.fit_transform(train_data_x)
    x_test_split = scaler.transform(x_test_split)
    test_data = scaler.transform(x_test)
    print("\ntrain shape", train_data_x.shape)

    # --------------------------------------------------------------------------------------------------------------
    # Fit model

    #Compute class weights:
    classes = np.unique(train_data_y['y'])
    class_weights = list(class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=train_data_y['y']))
    print("\nClass weights:\n", pd.DataFrame(class_weights))
    print("\nSamples per group before classification\n", train_data_y.groupby("y")["y"].count())
    class_weights_dict = dict(zip(range(len(class_weights)), class_weights))



    #Do prediction with svc
    model = SVC(C=1, class_weight=class_weights_dict, random_state=41, decision_function_shape='ovo')






    params = {}
    fit_params = {}
    params['C'] = [1]
    params['kernel'] = ['rbf', 'linear', 'poly']

    skf = StratifiedKFold(shuffle=True, n_splits=5, random_state=41)
    searcher = GridSearchCV(estimator=model, param_grid=params, scoring='balanced_accuracy', n_jobs=-1, refit=True, cv=skf, return_train_score=True, verbose=1)
    searcher.fit(train_data_x, train_data_y)

    # Best estimator
    print("Best estimator from GridSearch: {}".format(searcher.best_estimator_))
    print("Best alpha found: {}".format(searcher.best_params_))
    print("Best training-score with mse loss: {}".format(searcher.best_score_))
    results = pd.DataFrame(searcher.cv_results_)
    results.sort_values(by='rank_test_score', inplace=True)
    print(results[['params', 'mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score']].head(30))
    best_model = searcher.best_estimator_

    y_predict = best_model.predict(x_test_split)

    evaluation.evaluation_metrics(y_test_split, y_predict, "Test")
    visualize_prediction(x_test_split, y_test_split.to_numpy(), y_predict, "Test")


