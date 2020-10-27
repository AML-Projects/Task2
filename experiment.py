import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler, Normalizer

from source.classifier import Classifier
from source.data_sampler import DataSampling
from helpers.evaluation import one_hot_to_class, evaluation_metrics
from source.feature_extractor import FeatureExtractor


def output_submission(model, x_test, id_test):
    # make predictions
    predict = model.predict(x_test)
    predict = one_hot_to_class(predict, 3)
    # output
    output_csv = pd.concat([id_test, pd.Series(predict)], axis=1)
    output_csv.columns = ["id", "y"]
    pd.DataFrame.to_csv(output_csv, "./trainings/submit.csv", index=False)


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

    # --------------------------------------------------------------------------------------------------------------
    # Split
    x_train, x_validation, y_train, y_validation = \
        model_selection.train_test_split(train_data_x, train_data_y,
                                         test_size=0.2,
                                         stratify=train_data_y,
                                         random_state=41)

    print("\nTrain samples per group\n", y_train.groupby("y")["y"].count().values)
    print("\nValidation samples per group\n", y_validation.groupby("y")["y"].count().values)

    # reset all indexes
    x_train.reset_index(drop=True, inplace=True)
    x_validation.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_validation.reset_index(drop=True, inplace=True)

    # --------------------------------------------------------------------------------------------------------------
    # Scaling
    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_validation = scaler.transform(x_validation)
    x_test = scaler.transform(x_test)
    print("\ntrain shape", x_train.shape)

    # --------------------------------------------------------------------------------------------------------------
    # Feature Selection
    fs = FeatureExtractor("Nystroem")
    x_train = fs.fit_transform(x_train, y_train)
    x_validation = fs.transform(x_validation)
    x_test = fs.transform(x_test)

    print("\ntrain shape", x_train.shape)

    # visualize(x_train, y_train)

    # --------------------------------------------------------------------------------------------------------------
    # Sampling
    ds = DataSampling("None")
    x_train, y_train = ds.fit_resample(x_train, y_train)

    # --------------------------------------------------------------------------------------------------------------
    # Normalize samples?
    normalize_samples = False
    if normalize_samples:
        norm = Normalizer()
        x_train = norm.fit_transform(x_train)
        x_validation = norm.transform(x_validation)
        x_test = norm.transform(x_test)

    # --------------------------------------------------------------------------------------------------------------
    # Fit model
    clf = Classifier("SVC")
    model, results = clf.fit(X=x_train, y=y_train)

    # --------------------------------------------------------------------------------------------------------------
    # Evaluation
    best_model = model
    y_predict_train = best_model.predict(x_train)
    y_predict_validation = best_model.predict(x_validation)

    evaluation_metrics(y_train, y_predict_train, "Train", True)
    evaluation_metrics(y_validation, y_predict_validation, "Validation", True)

    if False:
        # train_data_y = tf.keras.utils.to_categorical(train_data_y, 3)
        best_model.fit(pd.DataFrame(x_train).append(pd.DataFrame(x_validation), ignore_index=True),
                       pd.DataFrame(y_train).append(pd.DataFrame(y_validation), ignore_index=True))
        output_submission(best_model, x_test, id_test)
