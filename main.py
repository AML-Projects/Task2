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
    parser.add_argument('--configuration', default='./configurations/ensemble.jsonc',
                        type=str, help="Environment and training configuration.")
    parser.add_argument('--workingdir', default=os.getcwd(), type=str,
                        help="Working directory (default: current directory).")
    parser.add_argument('--handin', default=False, type=argumenthelper.boolean_string,
                        help="If set to true, whole trainingset used for training")
    parser.add_argument('--hyperparamsearch', default=False, type=argumenthelper.boolean_string,
                        help="If set to true, will perform hyper parameter search, else it will only fit the given model")

    parser.add_argument('--ensemble_output', default=False, type=argumenthelper.boolean_string,
                        help="saves output of intermediate results if set to true")
    parser.add_argument('--ensemble_path',
                        default="/Users/sarahmorillo/PycharmProjects/AML/Task2/trainings/20201106-174337-e2,/Users/sarahmorillo/PycharmProjects/AML/Task2/trainings/20201029-211300-submission,trainings/20201105-135543-submission",
                        type=str,
                        help="string sep by ',' to indicate all the paths to dirs, where the individual preds. are stored")
    parser.add_argument('--ensemble_predict', default=True, type=argumenthelper.boolean_string,
                        help="will used prediction in ensemble_path to predict, if false no ensemble prediction is perfomed")
    #"trainings/20201106-164259-e1,trainings/20201106-174123-e3,trainings/20201106-174337-e2"

    parser.add_argument('--ensemble_handin', default=True, type=argumenthelper.boolean_string,
                        help="If set to true, will create submission.csv based on submission.csv of the individual classifiers, specified in ensebmle_path")

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
    Logcreator.info("Train-Data Shape: " + str(x_train.shape))
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
    elif args.ensemble_handin:
        engine.ensebmle_submission(args.ensemble_path, x_test_index=x_test.index)
    elif args.ensemble_predict:
        engine.ensemble_predict(args.ensemble_path, y_train_split=y_train_split, y_test_split=y_test_split)
    else:

        # Train
        classifier, x_test_split, x_train_split, y_train_split, search_results = engine.train(
                x_train_split=x_train_split, y_train_split=y_train_split, x_test_split=x_test_split, )

        # Predict
        engine.predict(clf=classifier, x_test_split=x_test_split, y_test_split=y_test_split,
                           x_train_split=x_train_split, y_train_split=y_train_split)

        if args.ensemble_output:
            Logcreator.info("Saving intermediate predictions for ensemble model")
            len_xtrain = len(x_train_split)
            len_xtest  = len(x_test_split)
            engine.save_output(clf =classifier, x= x_train_split, x_idx=list(range(0,len_xtrain)),
                                         filename="train_pred.csv")
            engine.save_output(clf=classifier, x=x_test_split, x_idx=list(range(len_xtrain,(len_xtrain +len_xtest))),
                                         filename="val_pred.csv")

        if args.handin:
            engine.output_submission(clf=classifier, x_test=x_test_split, x_test_index=x_test.index)



    end = time.time()
    Logcreator.info("Finished processing in %d [s]." % (end - start))
