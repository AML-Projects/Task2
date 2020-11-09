from helpers import evaluation
import numpy as np
import time
import argparse
import pandas as pd
from sklearn.utils import class_weight
import os
from source.configuration import Configuration
from logcreator.logcreator import Logcreator
from helpers import argumenthelper
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.preprocessing import RobustScaler
from imblearn.combine import SMOTEENN
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from matplotlib.colors import Normalize
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

# --------------------------------------------------------------------------------------------------------------
# Split
def perform_split(train_data_x, train_data_y, x_test, randState):
    Logcreator.info("Training with Train Test-Split of ratio 0.2 with random_state=", randState)
    if not args.handin:
        train_data_x, x_test_split, train_data_y, y_test_split = \
            model_selection.train_test_split(train_data_x, train_data_y, test_size=0.2, stratify=train_data_y,
                                             random_state=randState)
    else:
        x_test_split = x_test
        y_test_split = None
    return train_data_x, x_test_split, train_data_y, y_test_split


    # --------------------------------------------------------------------------------------------------------------
    # # Sampling
    #train_data_x, train_data_y = SMOTEENN(n_jobs=-1, random_state=41).fit_resample(train_data_x, train_data_y)
    #train_data_x, train_data_y = RandomOverSampler(random_state=41).fit_resample(train_data_x, train_data_y)
    #train_data_x, train_data_y = SMOTE(n_jobs=-1, random_state=41).fit_resample(train_data_x, train_data_y)
    #train_data_x, train_data_y = ADASYN(n_jobs=-1, random_state=41).fit_resample(train_data_x, train_data_y)
    #train_data_x, train_data_y = RandomUnderSampler(random_state=0).fit_resample(train_data_x, train_data_y)
    #train_data_x, train_data_y = ClusterCentroids(random_state=41).fit_resample(train_data_x, train_data_y)
    #train_data_x, train_data_y = AllKNN(sampling_strategy='not minority', n_jobs=-1).fit_resample(train_data_x, train_data_y)
    #train_data_x, train_data_y = SMOTETomek(n_jobs=-1, random_state=41).fit_resample(train_data_x, train_data_y)
    # Logcreator.info("\n Trainset samples per group\n", train_data_y.groupby("y")["y"].count().values)
    # if not args.handin:
    #     Logcreator.info("\n Testset samples per group\n", y_test_split.groupby("y")["y"].count().values)
    # --------------------------------------------------------------------------------------------------------------
    # Scaling
    # scaler = RobustScaler()
    # #
    # train_data_x = scaler.fit_transform(train_data_x)
    # x_test_split = scaler.transform(x_test_split)
    # Logcreator.info("\ntrain shape", train_data_x.shape)

    # --------------------------------------------------------------------------------------------------------------
    # Fit model

def fit_predict_model(train_data_x, x_test_split, train_data_y, y_test_split):
    do_gridsearch = False
    if do_gridsearch:
        #Compute class weights:
        classes = np.unique(train_data_y['y'])
        class_weights = list(class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=train_data_y['y']))
        Logcreator.info("\nClass weights:\n", pd.DataFrame(class_weights))
        Logcreator.info("\nSamples per group before classification\n", train_data_y.groupby("y")["y"].count())
        class_weights_dict = dict(zip(range(len(class_weights)), class_weights))

        #Do prediction with svc
        model = SVC(gamma='scale', class_weight='balanced', random_state=None, shrinking=True, decision_function_shape='ovo')


        kernel = ['rbf']

        #gamma_range = np.logspace(-9, 3, 5)
        #c_range = np.logspace(-2, 4, 7)
        gamma_range = ['scale']
        c_range = [1.1]
        param_grid = dict(kernel=kernel, C=c_range)

        skf = StratifiedKFold(shuffle=True, n_splits=10, random_state=41)
        searcher = GridSearchCV(estimator=model, param_grid=param_grid, scoring='balanced_accuracy', n_jobs=-1, refit=True, cv=skf, return_train_score=True, verbose=1)
        searcher.fit(train_data_x, train_data_y.values.ravel())

        # Best estimator
        Logcreator.info("Best estimator from GridSearch: {}".format(searcher.best_estimator_))
        Logcreator.info("Best parameters found: {}".format(searcher.best_params_))
        Logcreator.info("Best training-score with mse loss: {}".format(searcher.best_score_))
        results = pd.DataFrame(searcher.cv_results_)
        results.sort_values(by='rank_test_score', inplace=True)
        Logcreator.info(results[['params', 'mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score']].head(30))
        best_model = searcher.best_estimator_

        scores = searcher.cv_results_['mean_test_score'].reshape(len(c_range), len(gamma_range), len(kernel))

        # Draw heatmap of the validation accuracy as a function of gamma and C
        #
        # The score are encoded as colors with the hot colormap which varies from dark
        # red to bright yellow. As the most interesting scores are all located in the
        # 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
        # as to make it easier to visualize the small variations of score values in the
        # interesting range while not brutally collapsing all the low score values to
        # the same color.

        for i, kern in enumerate(kernel):
            plt.figure(figsize=(8, 6))
            plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
            plt.imshow(scores[:, :, i], interpolation='nearest', cmap=plt.cm.hot,
                       norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
            plt.xlabel('gamma')
            plt.ylabel('C')
            plt.colorbar()
            plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
            plt.yticks(np.arange(len(c_range)), c_range)
            plt.title('Validation accuracy ' + kern)
            plt.show()
            plt.savefig(os.path.join(Configuration.output_directory, "validation accuracy " + kern + "_HeatMap_C_gamma" + ".png"))
    else:

        kernel1 = RBF(length_scale=20, length_scale_bounds=(1e-05, 10000))
        best_model = SVC(C=1.1, class_weight='balanced', gamma='scale', kernel=kernel1, max_iter=-1, probability=False,
                    random_state=None,
                    shrinking=True, tol=0.001, verbose=False, decision_function_shape='ovo')
        best_model.fit(train_data_x, train_data_y.values.ravel())


    if args.handin:
        y_predict = best_model.predict(x_test_split)
        output_csv = pd.concat([pd.Series(x_test.index.values), pd.Series(y_predict.flatten())], axis=1)
        output_csv.columns = ["id", "y"]
        pd.DataFrame.to_csv(output_csv, os.path.join(Configuration.output_directory, 'submit.csv'), index=False)
    else:
        y_predict = best_model.predict(x_test_split)
        score = evaluation.evaluation_metrics(y_test_split, y_predict, "Test", False)
        #visualize_prediction(x_test_split, y_test_split.to_numpy(), y_predict, "Test")
        return score


if __name__ == '__main__':
    global config
    # Sample Config: --handin true --configuration D:\GitHub\AML\Task1\configurations\test.jsonc
    parser = argparse.ArgumentParser(
        description="Executes a training session.")
    parser.add_argument('--configuration', default='./configurations/test.jsonc',
                        type=str, help="Environment and training configuration.")
    parser.add_argument('--workingdir', default=os.getcwd(), type=str,
                        help="Working directory (default: current directory).")
    parser.add_argument('--handin', default=False, type=argumenthelper.boolean_string,
                        help="If set to true, whole trainingset used for training")
    parser.add_argument('--hyperparamsearch', default=False, type=argumenthelper.boolean_string,
                        help="If set to true, will perform hyper parameter search, else it will only fit the given model")

    args = argumenthelper.parse_args(parser)
    start = time.time()
    Configuration.initialize(args.configuration, args.workingdir)
    Logcreator.initialize()

    Logcreator.h1("Task 02 - MRI Desease classification")
    Logcreator.info("Environment: %s" % Configuration.get('environment.name'))

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

    Logcreator.info("Train-Data Shpae: " + str(train_data_x.shape))
    Logcreator.info("Test-Data Shape: " + str(x_test.shape))

    # --------------------------------------------------------------------------------------------------------------
    # Split
    # x_train, x_test, y_train, y_test = \
    #     model_selection.train_test_split(train_data_x, train_data_y,
    #                                      test_size=0.2,
    #                                      stratify=train_data_y,
    #                                      random_state=41)

    #print("\nTrain samples per group\n", train_data_x.groupby("y")["y"].count().values)
    Logcreator.info("\nValidation samples per group\n", train_data_y.groupby("y")["y"].count().values)

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

    nrSeeds = 15
    scores = []
    for index in range(40, nrSeeds+40):
        train_data_x, x_test_split, train_data_y, y_test_split = perform_split(train_data_x, train_data_y, x_test, index)
        score = fit_predict_model(train_data_x=train_data_x, x_test_split=x_test_split, train_data_y=train_data_y, y_test_split=y_test_split )
        scores.append(score)
    Logcreator.info("Scores are: ", scores)

    sum = sum(a for a in scores)
    Logcreator.info("The mean score over all seeds is: ", sum/nrSeeds)
