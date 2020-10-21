import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import xgboost as xgb
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, multilabel_confusion_matrix, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from tensorflow.python.keras.optimizer_v2.nadam import Nadam


def baseline_model():
    global nr_features
    model = Sequential()
    model.add(Dense(32, input_dim=nr_features, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
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
    class_weights = dict(zip(range(len(class_weights)), class_weights))
    model = baseline_model(nr_features)

    # model = KerasClassifier(build_fn=model)
    params = {"epochs": 60,
              "validation_data": (x_validation, y_validation),
              "class_weight": class_weights,
              "verbose": 1}
    model.fit(x, y, **params)

    return model


def xgb_classifier(x_train, y_train, weights):
    model = xgb.XGBClassifier(objective='multi:softmax', max_depth=1, num_class=3, random_state=41)

    model.fit(x_train, y_train.values.ravel(), sample_weight=weights)

    return model


def output_submission(model, x_test, id_test):
    # make predictions
    predict = model.predict(x_test)
    predict = one_hot_to_class(predict, 3)
    # output
    output_csv = pd.concat([id_test, pd.Series(predict)], axis=1)
    output_csv.columns = ["id", "y"]
    pd.DataFrame.to_csv(output_csv, "./trainings/submit.csv", index=False)


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    https://scikit-learn.org/0.21/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # check if arrays are one hot encoded
    y_true = one_hot_to_class(y_true, len(np.unique(classes)))
    y_pred = one_hot_to_class(y_pred, len(np.unique(classes)))

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def one_hot_to_class(y, num_classes):
    if len(y.shape) > 1 and y.shape[1] == num_classes:
        y = np.argmax(y, axis=1)
    return y


def evaluation_metrics(y_true, y_predicted, text):
    print("\n---------", text, "---------\n")
    # confusion matrix
    plot_confusion_matrix(y_true, y_predicted, classes=classes,
                          title=text + ' - Confusion matrix', normalize=False)
    plt.show()

    # convert from one hot encoding to class labels if needed
    y_true = one_hot_to_class(y_true, num_classes=number_classes)
    y_predicted = one_hot_to_class(y_predicted, num_classes=number_classes)

    # report
    print(text, 'report')
    print(classification_report(y_true, y_predicted))

    # balanced accuracy score
    score = balanced_accuracy_score(y_true, y_predicted)
    print("bas_score on", text, "split:", score)


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
    print(x_train.shape)

    # --------------------------------------------------------------------------------------------------------------
    # Compute class weights
    classes = np.unique(y_train['y'])
    number_classes = len(classes)
    class_weights = list(class_weight.compute_class_weight(class_weight='balanced',
                                                           classes=classes,
                                                           y=y_train['y']))

    weights = y_train.copy()
    for i in range(0, len(class_weights)):
        weights.loc[weights.y == i] = class_weights[i]

    nr_features = x_train.shape[1]

    # --------------------------------------------------------------------------------------------------------------
    # Fit model
    xgb_on = False
    if xgb_on:
        model = xgb_classifier(x_train, y_train, weights)
    else:
        # to on hot
        y_train = tf.keras.utils.to_categorical(y_train, 3)
        y_validation = tf.keras.utils.to_categorical(y_validation, 3)
        model = neural_network(x_train, y_train, class_weights, nr_features)

    # --------------------------------------------------------------------------------------------------------------
    # Evaluation
    best_model = model
    y_predict_train = best_model.predict(x_train)
    y_predict_validation = best_model.predict(x_validation)

    evaluation_metrics(y_train, y_predict_train, "Train")
    evaluation_metrics(y_validation, y_predict_validation, "Validation")

    if True:
        train_data_y = tf.keras.utils.to_categorical(train_data_y, 3)
        best_model.fit(train_data_x, train_data_y)
        output_submission(best_model, x_test, id_test)
