"""
Runs trainings and predictions
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Sarah Morillo'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, sleonardo@student.ethz.ch"

import itertools
import os

import pandas as pd
from numpy.random import seed
from sklearn.preprocessing import Normalizer
from tensorflow.python.framework.random_seed import set_random_seed

from helpers import evaluation
from helpers.visualize import visualize_prediction
from logcreator.logcreator import Logcreator
from source.classifier import Classifier
from source.configuration import Configuration
from source.data_sampler import DataSampling
from source.feature_adder import FeatureAdder
from source.feature_extractor import FeatureExtractor
from source.feature_transformer import FeatureTransformer
from source.scaler import Scaler
import re
import glob

class Engine:
    def __init__(self, fix_random_seeds=True):
        Logcreator.info("Training initialized")
        if fix_random_seeds:
            seed(1)
            set_random_seed(1)

    def search(self, x_train_split, y_train_split, x_test_split, y_test_split):
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        data_sampler_par_list = self.get_search_list('search.data_sampler')
        scaler_par_list = self.get_search_list('search.scaler')
        data_feature_adder_par_list = self.get_search_list('search.feature_adder')
        classifier_par_list = self.get_search_list('search.classifier')

        number_of_loops = len(data_sampler_par_list) * len(scaler_par_list) * len(data_feature_adder_par_list) * len(
            classifier_par_list)

        Logcreator.h1("Number of loops:", number_of_loops)
        loop_counter = 0

        # prepare out columns names
        columns_out = ["Loop_counter", "BAS Score Test"]
        columns_out.extend(['params', 'mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score'])
        # combine all keys
        columns_out.extend(['data_sampler_' + s for s in list(data_sampler_par_list[0].keys())])
        columns_out.extend(['scaler_' + s for s in list(scaler_par_list[0].keys())])
        columns_out.extend(['feature_adder_' + s for s in list(data_feature_adder_par_list[0].keys())])
        columns_out.extend(['classifier_' + s for s in list(classifier_par_list[0].keys())])

        # create output dataframe
        results_out = pd.DataFrame(columns=columns_out)
        pd.DataFrame.to_csv(results_out, os.path.join(Configuration.output_directory, 'search_results.csv'),
                            index=False)

        try:
            for data_sampler_data in data_sampler_par_list:

                ds = DataSampling(**data_sampler_data)
                x_train_split_ds, y_train_split_ds = ds.fit_resample(x_train_split, y_train_split)
                x_test_split_ds = x_test_split
                Logcreator.info("\n Trainset samples per group\n", y_train_split_ds.groupby("y")["y"].count().values)

                for scaler_data in scaler_par_list:

                    scaler = Scaler(**scaler_data)
                    x_train_split_sc, y_train_split_sc, x_test_split_sc = scaler.transform_custom(
                        x_train=x_train_split_ds,
                        y_train=y_train_split_ds,
                        x_test=x_test_split_ds)

                    for feature_adder_data in data_feature_adder_par_list:

                        feature_adder = FeatureAdder(**feature_adder_data)
                        feature_adder.fit(x_train_split_sc, y_train_split_sc)
                        x_train_split_adder = feature_adder.transform(x_train_split_sc)
                        x_test_split_adder = feature_adder.transform(x_test_split_sc)
                        y_train_split_adder = y_train_split_sc

                        for classifier_data in classifier_par_list:
                            Logcreator.info("\n--------------------------------------")
                            Logcreator.info("Iteration", loop_counter)
                            Logcreator.info("Data_sampler", data_sampler_data)
                            Logcreator.info("Scaler", scaler_data)
                            Logcreator.info("Feature Adder", feature_adder_data)
                            Logcreator.info("Classifier", classifier_data)
                            Logcreator.info("\n----------------------------------------")

                            # Train classifier
                            clf = Classifier(**classifier_data)
                            best_model = clf.fit(X=x_train_split_adder, y=y_train_split_adder)
                            search_results = clf.getFitResults()

                            # Predict test data
                            y_predict_test = best_model.predict(x_test_split_adder)
                            score_test = evaluation.evaluation_metrics(y_test_split, y_predict_test, "Test", False)
                            Logcreator.info("BAS Score achieved on test set: {}".format(score_test))
                            # visualize_prediction(x_test_split, y_test_split, y_predict_test, "Test")

                            output = pd.DataFrame()

                            nrOfOutputRows = 5 if len(search_results) > 5 else len(search_results)
                            for i in range(0,
                                           nrOfOutputRows):  # append multiple rows of the grid search result, not just the best
                                # update output
                                output_row = list()
                                output_row.append(loop_counter)
                                output_row.append(score_test)
                                output_row.extend(search_results[
                                                      ['params', 'mean_test_score', 'std_test_score',
                                                       'mean_train_score',
                                                       'std_train_score']].iloc[i])

                                output_row.extend(list(data_sampler_data.values()))
                                output_row.extend(list(scaler_data.values()))
                                output_row.extend(list(feature_adder_data.values()))
                                output_row.extend(list(classifier_data.values()))
                                output = output.append(pd.DataFrame(output_row, index=results_out.columns).T)

                            # Write to csv
                            pd.DataFrame.to_csv(output,
                                                os.path.join(Configuration.output_directory, 'search_results.csv'),
                                                index=False, mode='a', header=False)
                            # Increase loop counter
                            loop_counter = loop_counter + 1

        finally:
            Logcreator.info("Search finished")

    def get_search_list(self, config_name):
        param_dict = self.get_search_params(config_name)
        keys, values = zip(*param_dict.items())

        search_list = list()
        # probably there is an easier way to do this
        for instance in itertools.product(*values):
            d = dict(zip(keys, instance))
            search_list.append(d)

        return search_list

    def get_search_params(self, config_name):
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

    def train(self, x_train_split, y_train_split, x_test_split):
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        # --------------------------------------------------------------------------------------------------------------
        # Sampling
        ds = DataSampling(sampling_method=Configuration.get('data_sampler.sampling_method'))
        # ds_dict = self.get_serach_params('data_sampler')
        # ds = DataSampling(ds)
        x_train_split, y_train_split = ds.fit_resample(x_train_split, y_train_split)
        Logcreator.info("\n Trainset samples per group\n", y_train_split.groupby("y")["y"].count().values)

        # --------------------------------------------------------------------------------------------------------------
        # Scaling
        scaler = Scaler(name=Configuration.get('scaler.name'))
        x_train_split, y_train_split, x_test_split = scaler.transform_custom(x_train=x_train_split,
                                                                             y_train=y_train_split,
                                                                             x_test=x_test_split)

        # --------------------------------------------------------------------------------------------------------------
        # Feature Selection
        featureselection = False
        if featureselection:
            fs_dict = self.get_search_params('feature_selector')
            fs = FeatureExtractor(**fs_dict)

            x_train_split = fs.fit_transform(x_train_split, y_train_split)
            x_test_split = fs.transform(x_test_split)

            Logcreator.info("\ntrain shape", x_train_split.shape)

        # visualize(x_train, y_train)

        # --------------------------------------------------------------------------------------------------------------
        # Feature transformation
        ft = FeatureTransformer(method="None")
        x_train_split = ft.fit_transform(x_train_split)
        x_test_split = ft.transform(x_test_split)

        # --------------------------------------------------------------------------------------------------------------
        # Feature adder
        fadder = FeatureAdder(clustering_on=Configuration.get('feature_adder.clustering_on'),
                              n_clusters=Configuration.get('feature_adder.n_clusters'),
                              custom_on=Configuration.get('feature_adder.custom_on'),
                              auto_encoder_on=Configuration.get('feature_adder.auto_encoder_on'),
                              n_encoder_features=Configuration.get('feature_adder.n_encoder_features'),
                              encoder_path=Configuration.get('feature_adder.encoder_path'),
                              lda_on=Configuration.get('feature_adder.lda_on'),
                              lda_shrinkage=Configuration.get('feature_adder.lda_shrinkage'),
                              replace_features=Configuration.get('feature_adder.replace_features'))
        fadder.fit(x_train_split, y_train_split)
        x_train_split = fadder.transform(x_train_split)
        x_test_split = fadder.transform(x_test_split)

        # --------------------------------------------------------------------------------------------------------------
        # Normalize samples?
        normalize_samples = False
        if normalize_samples:
            norm = Normalizer()
            x_train_split = norm.fit_transform(x_train_split)
            x_test_split = norm.transform(x_test_split)

        # --------------------------------------------------------------------------------------------------------------
        # Fit model
        clf = Classifier(classifier=Configuration.get('classifier.classifier'),
                         random_search=Configuration.get('classifier.random_search'))
        best_model = clf.fit(X=x_train_split, y=y_train_split)
        results = clf.getFitResults() # auskomment... für SVM

        return best_model, x_test_split, x_train_split, y_train_split, results

    def predict(self, clf, x_test_split, y_test_split, x_train_split, y_train_split):
        y_predict_train = clf.predict(x_train_split)

        evaluation.evaluation_metrics(y_train_split, y_predict_train, "Train")
        # visualize_prediction(x_train_split, y_train_split, y_predict_train, "Train")

        if y_test_split is not None:
            y_predict_test = clf.predict(x_test_split)
            evaluation.evaluation_metrics(y_test_split, y_predict_test, "Test")
            # visualize_prediction(x_test_split, y_test_split, y_predict_test, "Test")

    def output_submission(self, clf, x_test, x_test_index, filename="submit.csv"):
        predicted_values = clf.predict(x_test)
        classes = [0, 1, 2]
        y_pred = evaluation.one_hot_to_class(predicted_values, len(classes))
        output_csv = pd.concat([pd.Series(x_test_index.values), pd.Series(y_pred.flatten())], axis=1)
        output_csv.columns = ["id", "y"]
        pd.DataFrame.to_csv(output_csv, os.path.join(Configuration.output_directory, filename), index=False)

    def save_output(self, clf, x, x_idx, filename):
        predicted_values = clf.predict(x)
        classes = [0,1,2]
        y_pred = evaluation.one_hot_to_class(predicted_values, len(classes))
        output_csv = pd.concat([pd.Series(x_idx), pd.Series(y_pred.flatten())], axis=1)
        output_csv.columns = ["id", "y"]
        pd.DataFrame.to_csv(output_csv, os.path.join(Configuration.output_directory, filename), index=False)

    def ensemble_predict(self, paths, y_train_split, y_test_split):
        Logcreator.info("Ensemble predict")
        # remove spaces
        paths = paths.replace(" ", "")
        paths = paths.split(",")
        cwd = os.getcwd()
        predicted_values_train = pd.DataFrame()
        predicted_values_val = pd.DataFrame()
        for path in paths:
            if os.path.exists(path):
                Logcreator.info("Reading in prediction from {}".format(path))
                os.chdir(path)
                train_pred = pd.read_csv("train_pred.csv", index_col=0)
                predicted_values_train = pd.concat([predicted_values_train,train_pred], axis=1)
                val_pred = pd.read_csv("val_pred.csv", index_col=0)
                predicted_values_val = pd.concat([predicted_values_val, val_pred], axis=1)
                os.chdir(cwd)

        y_predict_train = predicted_values_train.mode(axis=1)[0]
        y_predict_val = predicted_values_val.mode(axis=1)[0]
        evaluation.evaluation_metrics(y_train_split, y_predict_train, "Train")
        if y_test_split is not None:
            evaluation.evaluation_metrics(y_test_split, y_predict_val, "Test")


    def ensebmle_submission(self, paths, x_test_index, filename="submit.csv"):
        # remove spaces
        paths = paths.replace(" ", "")
        paths = paths.split(",")
        cwd = os.getcwd()
        predicted_values = pd.DataFrame()
        for path in paths:
            if os.path.exists(path):
                Logcreator.info("Reading in prediction from {}".format(path))
                os.chdir(path)
                #dir_name = path.split("/")[-1]
                #prefix = re.findall("\d{8}-\d{6}-", dir_name)[0]
                tmp = glob.glob("*submit.csv")
                if len(tmp) > 0:
                    submision_file_name = glob.glob("*submit.csv")[0]
                    if os.path.exists(submision_file_name):
                        Logcreator.info("Reading in {}".format(submision_file_name))
                        predictions = pd.read_csv(submision_file_name, index_col=0)
                        predicted_values = pd.concat([predicted_values, predictions], axis=1)
                else:
                    Logcreator.info("No submission file found in {}".format(path))
                os.chdir(cwd)

        prediction = predicted_values.mode(axis=1)[0]
        output_csv = pd.concat([pd.Series(x_test_index.values), pd.Series(prediction)], axis=1).astype(int)
        output_csv.columns = ["id", "y"]
        pd.DataFrame.to_csv(output_csv, os.path.join(Configuration.output_directory, filename), index=False)