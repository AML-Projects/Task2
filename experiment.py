import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import model_selection
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight


def output_testdata(model, x_test, id_test):
    # make predictions
    predict = model.predict(x_test)
    # output
    output_csv = pd.concat([id_test, pd.Series(predict)], axis=1)
    output_csv.columns = ["id", "y"]
    pd.DataFrame.to_csv(output_csv, "./data/submit.csv", index=False)


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

    # --------------------------------------------------------------------------------------------------------------
    # Compute class weights
    class_weights = list(class_weight.compute_class_weight(class_weight='balanced',
                                                           classes=np.unique(y_train['y']),
                                                           y=y_train['y']))

    weights = y_train.copy()
    for i in range(0, len(class_weights)):
        weights.loc[weights.y == i] = class_weights[i]

    # --------------------------------------------------------------------------------------------------------------
    # Fit model
    model = xgb.XGBClassifier(random_state=41)

    model.fit(x_train, y_train.values.ravel(), sample_weight=weights)

    best_model = model
    predict_train = best_model.predict(x_train)
    predict_validation = best_model.predict(x_validation)

    score = balanced_accuracy_score(y_train, predict_train)
    print("bas_score on train split: ", score)

    score = balanced_accuracy_score(y_validation, predict_validation)
    print("bas_score on validation split: ", score)

    if False:
        output_testdata(best_model, x_test, id_test)
