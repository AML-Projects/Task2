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
from matplotlib.colors import Normalize

from source.classifier import Classifier
from source.data_sampler import DataSampling

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

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
    x_test.reset_index(drop=True, inplace=True)

    pca = PCA(n_components=100)
    components = pca.fit_transform(train_data_x)

    for i in range(1,1):

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
    handin = False
    if not handin:
        train_data_x, x_test_split, train_data_y, y_test_split = \
        model_selection.train_test_split(train_data_x, train_data_y, test_size=0.2, stratify=train_data_y, random_state=41)
    else:
        x_test_split = x_test


    # --------------------------------------------------------------------------------------------------------------
    # # Sampling
    train_data_x, train_data_y = SMOTEENN(n_jobs=-1, random_state=41).fit_resample(train_data_x, train_data_y)
    #train_data_x, train_data_y = RandomOverSampler(random_state=41).fit_resample(train_data_x, train_data_y)
    #train_data_x, train_data_y = SMOTE(n_jobs=-1, random_state=41).fit_resample(train_data_x, train_data_y)
    #train_data_x, train_data_y = ADASYN(n_jobs=-1, random_state=41).fit_resample(train_data_x, train_data_y)
    #train_data_x, train_data_y = RandomUnderSampler(random_state=0).fit_resample(train_data_x, train_data_y)
    #train_data_x, train_data_y = ClusterCentroids(random_state=41).fit_resample(train_data_x, train_data_y)
    #train_data_x, train_data_y = AllKNN(sampling_strategy='not minority', n_jobs=-1).fit_resample(train_data_x, train_data_y)
    #train_data_x, train_data_y = SMOTETomek(n_jobs=-1, random_state=41).fit_resample(train_data_x, train_data_y)
    print("\n Trainset samples per group\n", train_data_y.groupby("y")["y"].count().values)
    if not handin:
        print("\n Testset samples per group\n", y_test_split.groupby("y")["y"].count().values)
    # --------------------------------------------------------------------------------------------------------------
    # Scaling
    scaler = RobustScaler()
    #
    train_data_x = scaler.fit_transform(train_data_x)
    x_test_split = scaler.transform(x_test_split)
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
    model = SVC(class_weight=class_weights_dict, random_state=41, decision_function_shape='ovo')

    kernel = ['rbf', 'linear', 'poly']
    gamma_range = np.logspace(-9, 3,13)
    c_range = np.logspace(-2, 10, 13)
    param_grid = dict(gamma=gamma_range, kernel=kernel, C=c_range)

    skf = StratifiedKFold(shuffle=True, n_splits=5, random_state=41)
    searcher = GridSearchCV(estimator=model, param_grid=param_grid, scoring='balanced_accuracy', n_jobs=-1, refit=True, cv=skf, return_train_score=True, verbose=1)
    searcher.fit(train_data_x, train_data_y)

    # Best estimator
    print("Best estimator from GridSearch: {}".format(searcher.best_estimator_))
    print("Best parameters found: {}".format(searcher.best_params_))
    print("Best training-score with mse loss: {}".format(searcher.best_score_))
    results = pd.DataFrame(searcher.cv_results_)
    results.sort_values(by='rank_test_score', inplace=True)
    print(results[['params', 'mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score']].head(30))
    best_model = searcher.best_estimator_



    scores = searcher.cv_results_['mean_test_score'].reshape(len(c_range),
                                                         len(gamma_range))

    # Draw heatmap of the validation accuracy as a function of gamma and C
    #
    # The score are encoded as colors with the hot colormap which varies from dark
    # red to bright yellow. As the most interesting scores are all located in the
    # 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
    # as to make it easier to visualize the small variations of score values in the
    # interesting range while not brutally collapsing all the low score values to
    # the same color.

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(c_range)), c_range)
    plt.title('Validation accuracy')
    plt.show()

    if handin:
        y_predict = best_model.predict(x_test_split)
        output_csv = pd.concat([pd.Series(x_test.index.values), pd.Series(y_predict.flatten())], axis=1)
        output_csv.columns = ["id", "y"]
        pd.DataFrame.to_csv(output_csv, os.path.join("D:\\temp", 'submit.csv'), index=False)
    else:
        y_predict = best_model.predict(x_test_split)
        evaluation.evaluation_metrics(y_test_split, y_predict, "Test")
        visualize_prediction(x_test_split, y_test_split.to_numpy(), y_predict, "Test")


