"""
Runs trainings and predictions
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Sarah Morillo'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, sleonardo@student.ethz.ch"

import itertools
import os

import pandas as pd
from numpy.random import seed
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from tensorflow.python.framework.random_seed import set_random_seed

from logcreator.logcreator import Logcreator
from source import evaluation
from source.autoencoder import AutoEncoder
from source.classifier import Classifier
from source.configuration import Configuration
from source.data_sampler import DataSampling
from source.feature_extractor import FeatureExtractor
from source.visualize import visualize_prediction, visualize_true_labels


class Engine:
    def __init__(self, fix_random_seeds=True):
        Logcreator.info("Training initialized")
        if fix_random_seeds:
            seed(1)
            set_random_seed(1)

    def search(self, x_train, y_train, x_test):

        imputer_par_list = self.get_serach_list('search.imputer')
        outlier_par_list = self.get_serach_list('search.outlier')
        feature_selector_par_list = self.get_serach_list('search.feature_selector')
        normalizer_par_list = self.get_serach_list('search.normalizer')
        regression_par_list = self.get_serach_list('search.regression')

        number_of_loops = len(imputer_par_list) * len(outlier_par_list) * len(feature_selector_par_list) \
                          * len(normalizer_par_list) * len(regression_par_list)

        Logcreator.h1("Number of loops:", number_of_loops)
        loop_counter = 0

        # prepare out columns names
        columns_out = ["Loop_counter", "R2 Score Test", "R2 Score Training"]
        columns_out.extend(['params', 'mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score'])
        # combine all keys
        columns_out.extend(['imputer_' + s for s in list(imputer_par_list[0].keys())])
        columns_out.extend(['outlier_' + s for s in list(outlier_par_list[0].keys())])
        columns_out.extend(['feature_selector_' + s for s in list(feature_selector_par_list[0].keys())])
        columns_out.extend(['normalizer_' + s for s in list(normalizer_par_list[0].keys())])
        columns_out.extend(['regression_' + s for s in list(regression_par_list[0].keys())])

        # create output dataframe
        results_out = pd.DataFrame(columns=columns_out)
        pd.DataFrame.to_csv(results_out, os.path.join(Configuration.output_directory, 'search_results.csv'),
                            index=False)

        """
        try:
            for imp_data in imputer_par_list:

                imputer = Imputer(**imp_data)
                x_train_imp, y_train_imp, x_test_imp = imputer.transform_custom(x_train=x_train,
                                                                                y_train=y_train,
                                                                                x_test=x_test)

                for out_data in outlier_par_list:

                    outlier = Outliers(**out_data)
                    x_train_out, y_train_out, x_test_out = outlier.transform_custom(x_train=x_train_imp,
                                                                                    y_train=y_train_imp,
                                                                                    x_test=x_test_imp)

                    for feature_selector_data in feature_selector_par_list:

                        feature_selector = FeatureSelector(**feature_selector_data)

                        x_train_fs, y_train_fs, x_test_fs = feature_selector.transform_custom(x_train=x_train_out,
                                                                                              y_train=y_train_out,
                                                                                              x_test=x_test_out)

                        for normalizer_data in normalizer_par_list:

                            normalizer = Normalizer(**normalizer_data)

                            x_train_norm, y_train_norm, x_test_norm = normalizer.transform_custom(x_train=x_train_fs,
                                                                                                  y_train=y_train_fs,
                                                                                                  x_test=x_test_fs)

                            for regression_data in regression_par_list:
                                # TODO clean up output of current parameters
                                Logcreator.info("\n--------------------------------------")
                                Logcreator.info("Iteration", loop_counter)
                                Logcreator.info("imputer", imp_data)
                                Logcreator.info("outlier", out_data)
                                Logcreator.info("feature_selector", feature_selector_data)
                                Logcreator.info("normalizer", normalizer_data)
                                Logcreator.info("regression", regression_data)
                                Logcreator.info("\n----------------------------------------")

                                regressor = Regression(**regression_data)
                                best_model, x_test_split, y_test_split, x_train_split, y_train_split, search_results = \
                                    regressor.fit_predict(
                                        x_train=x_train_norm, y_train=y_train_norm,
                                        x_test=x_test_norm, handin=False)

                                predicted_values = best_model.predict(x_train_split)
                                score_train = r2_score(y_true=y_train_split, y_pred=predicted_values)
                                Logcreator.info("R2 Score achieved on training set: {}".format(score_train))

                                predicted_values = best_model.predict(x_test_split)
                                score_test = r2_score(y_true=y_test_split, y_pred=predicted_values)
                                Logcreator.info("R2 Score achieved on test set: {}".format(score_test))

                                output = pd.DataFrame()
                                for i in range(0,
                                               5):  # append multiple rows of the grid search result, not just the best
                                    # update output
                                    # TODO not so nice because we only take the values, so the order has to be correct;
                                    #  Maybe converte everything to one dictionary and then append the dictionary to the pandas dataframe;
                                    #  But works for now as long as the order is correct
                                    output_row = list()
                                    output_row.append(loop_counter)
                                    output_row.append(score_test)
                                    output_row.append(score_train)
                                    output_row.extend(search_results[
                                                          ['params', 'mean_test_score', 'std_test_score',
                                                           'mean_train_score',
                                                           'std_train_score']].iloc[i])

                                    output_row.extend(list(imp_data.values()))
                                    output_row.extend(list(out_data.values()))
                                    output_row.extend(list(feature_selector_data.values()))
                                    output_row.extend(list(normalizer_data.values()))
                                    output_row.extend(list(regression_data.values()))
                                    output = output.append(pd.DataFrame(output_row, index=results_out.columns).T)

                                # Write to csv
                                pd.DataFrame.to_csv(output,
                                                    os.path.join(Configuration.output_directory, 'search_results.csv'),
                                                    index=False, mode='a', header=False)
                                # Increase loop counter
                                loop_counter = loop_counter + 1
            
        finally:
            Logcreator.info("Search finished")
        """

    def get_serach_list(self, config_name):
        param_dict = self.get_serach_params(config_name)
        keys, values = zip(*param_dict.items())

        search_list = list()
        # probably there is an easier way to do this
        for instance in itertools.product(*values):
            d = dict(zip(keys, instance))
            search_list.append(d)

        return search_list

    def get_serach_params(self, config_name):
        """
        Not so nice to parse the data from the file to a dictionary...
        Parameters have to have the exact name of the actual class parameter!
        Parameters have to be alphabetically ordered in the config file!
        Parameters can't be named count and index!
        """

        config = Configuration.get(config_name)
        attribute_names = [a for a in dir(config) if not a.startswith('_')]
        # count and index is somehow also an attribute of the config object
        if 'count' in attribute_names:
            attribute_names.remove('count')
        if 'index' in attribute_names:
            attribute_names.remove('index')

        param_dict = dict()
        for name, element in zip(attribute_names, config):
            param_dict[name] = element

        return param_dict

    def train(self, x_train, y_train, x_test):
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        # --------------------------------------------------------------------------------------------------------------
        # Split
        x_train_split, x_validation_split, y_train_split, y_validation_split = \
            model_selection.train_test_split(x_train, y_train,
                                             test_size=0.2,
                                             stratify=y_train,
                                             shuffle=True,
                                             random_state=41)

        Logcreator.info("\nTrain samples per group\n", y_train_split.groupby("y")["y"].count().values)
        Logcreator.info("\nValidation samples per group\n", y_validation_split.groupby("y")["y"].count().values)

        # reset all indexes
        x_train_split.reset_index(drop=True, inplace=True)
        x_validation_split.reset_index(drop=True, inplace=True)
        y_train_split.reset_index(drop=True, inplace=True)
        y_validation_split.reset_index(drop=True, inplace=True)

        # --------------------------------------------------------------------------------------------------------------
        # Scaling
        scaler = StandardScaler()

        x_train_split = scaler.fit_transform(x_train_split)
        x_validation_split = scaler.transform(x_validation_split)
        x_test = scaler.transform(x_test)
        Logcreator.info("\ntrain shape", x_train_split.shape)

        # --------------------------------------------------------------------------------------------------------------
        # Feature Extraction
        # TODO move to feature extraction
        auto_encoder = False
        if auto_encoder:
            scaler = MinMaxScaler(feature_range=(-1, 1))

            x_train_split = scaler.fit_transform(x_train_split)
            x_validation_split = scaler.transform(x_validation_split)
            x_test = scaler.transform(x_test)

            ae = AutoEncoder()
            x_train_split = ae.fit_transform(x_train_split, y_train_split)
            x_validation_split = ae.transform(x_validation_split)
            x_test = ae.transform(x_test)

            visualize_true_labels(x_train_split, y_train_split, "Train")
            visualize_true_labels(x_validation_split, y_validation_split, "Validation")

        # --------------------------------------------------------------------------------------------------------------
        # Feature Selection
        fs_dict = self.get_serach_params('feature_selector')
        fs = FeatureExtractor(**fs_dict)

        x_train_split = fs.fit_transform(x_train_split, y_train_split)
        x_validation_split = fs.transform(x_validation_split)
        x_test = fs.transform(x_test)

        Logcreator.info("\ntrain shape", x_train_split.shape)

        # visualize(x_train, y_train)

        # --------------------------------------------------------------------------------------------------------------
        # Sampling
        ds_dict = self.get_serach_params('data_sampler')
        ds = DataSampling(**ds_dict)
        x_train_split, y_train_split = ds.fit_resample(x_train_split, y_train_split)

        # --------------------------------------------------------------------------------------------------------------
        # Normalize samples?
        normalize_samples = False
        if normalize_samples:
            norm = Normalizer()
            x_train_split = norm.fit_transform(x_train_split)
            x_validation_split = norm.transform(x_validation_split)
            x_test = norm.transform(x_test)

        # --------------------------------------------------------------------------------------------------------------
        # Fit model
        clf_dict = self.get_serach_params('classifier')
        clf = Classifier(**clf_dict)
        best_model = clf.fit(X=x_train_split, y=y_train_split)
        results = clf.getFitResults()

        return best_model, x_validation_split, y_validation_split, x_train_split, y_train_split, x_test, results

    def predict(self, clf, x_test_split, y_test_split, x_train_split, y_train_split):
        y_predict_train = clf.predict(x_train_split)

        evaluation.evaluation_metrics(y_train_split, y_predict_train, "Train")
        visualize_prediction(x_train_split, y_train_split, y_predict_train, "Train")

        if y_test_split is not None:
            y_predict_validation = clf.predict(x_test_split)

            evaluation.evaluation_metrics(y_test_split, y_predict_validation, "Validation")
            visualize_prediction(x_test_split, y_test_split, y_predict_validation, "Validation")

    def output_submission(self, clf, x_test, x_test_index):
        predicted_values = clf.predict(x_test)
        output_csv = pd.concat([pd.Series(x_test_index.values), pd.Series(predicted_values.flatten())], axis=1)
        output_csv.columns = ["id", "y"]
        pd.DataFrame.to_csv(output_csv, os.path.join(Configuration.output_directory, 'submit.csv'), index=False)
