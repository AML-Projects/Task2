"""
Main class
"""

__author__ = 'Andreas Kaufmann'
__email__ = "ankaufmann@student.ethz.ch"

import os
import time
import argparse
import pandas as pd
from sklearn import model_selection
from source.configuration import Configuration
from source.engine import Engine
from logcreator.logcreator import Logcreator
from helpers import argumenthelper

if __name__ == "__main__":
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

    # Load training data
    x_train = pd.read_csv("./data/X_train.csv", index_col=0)
    Logcreator.info("Shape of training_samples: {}".format(x_train.shape))
    Logcreator.info(x_train.head())

    y_train = pd.read_csv("./data/y_train.csv", index_col=0)
    Logcreator.info("Shape of training labels: {}".format(y_train.shape))
    Logcreator.info(y_train.head())

    x_test = pd.read_csv("./data/X_test.csv", index_col=0)
    Logcreator.info("Shape of test samples: {}".format(x_test.shape))
    Logcreator.info(x_test.head())

    #Prepare data for training
    Logcreator.info("Train-Data Shpae: " + str(x_train.shape))
    Logcreator.info("Test-Data Shape: " + str(x_test.shape))
    Logcreator.info("\nValidation samples per group\n", y_train.groupby("y")["y"].count().values)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    #Split data into training and testdata
    if args.handin:
        x_train_split = x_train
        y_train_split = y_train
        x_test_split = x_test
        y_test_split = None
    else:
        x_train_split, x_test_split, y_train_split, y_test_split = \
            model_selection.train_test_split(x_train, y_train, test_size=0.2, stratify=y_train, shuffle=True, random_state=41)
    Logcreator.info("\nTrain samples per group\n", y_train_split.groupby("y")["y"].count().values)
    if y_test_split is not None:
        Logcreator.info("\nTest samples per group\n", y_test_split.groupby("y")["y"].count().values)

    # reset all indexes
    x_train_split.reset_index(drop=True, inplace=True)
    y_train_split.reset_index(drop=True, inplace=True)
    x_test_split.reset_index(drop=True, inplace=True)
    if y_test_split is not None:
        y_test_split.reset_index(drop=True, inplace=True)

    engine = Engine()
    # Hyperparamter Search
    if args.hyperparamsearch:
        engine.search(x_train_split, y_train_split, x_test_split, y_test_split)
    else:
        # Train
        classifier, x_test_split, x_train_split, y_train_split, search_results = engine.train(
            x_train_split=x_train_split, y_train_split=y_train_split, x_test_split=x_test_split, )

        # Predict
        engine.predict(clf=classifier, x_test_split=x_test_split, y_test_split=y_test_split,
                       x_train_split=x_train_split, y_train_split=y_train_split)

        if args.handin:
            engine.output_submission(clf=classifier, x_test=x_test_split, x_test_index=x_test.index)

    end = time.time()
    Logcreator.info("Finished processing in %d [s]." % (end - start))
