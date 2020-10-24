import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import balanced_accuracy_score, multilabel_confusion_matrix, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, Normalizer

from source.classifier import Classifier
from source.data_sampler import DataSampling
from source.feature_selector import FeatureSelector


def plot_individual_cm(y_true, y_predicted):
    cm = multilabel_confusion_matrix(y_true, y_predicted, labels=[0, 1, 2])

    for i in range(cm.shape[0]):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm[i],
                                      display_labels=[i, "rest"])
        disp = disp.plot()
    plt.show()


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
    classes = np.unique(y_true)
    # confusion matrix
    plot_confusion_matrix(y_true, y_predicted, classes=classes,
                          title=text + ' - Confusion matrix', normalize=True)
    plt.show()

    # convert from one hot encoding to class labels if needed
    number_classes = len(classes)
    y_true = one_hot_to_class(y_true, num_classes=number_classes)
    y_predicted = one_hot_to_class(y_predicted, num_classes=number_classes)

    # plot_individual_cm(y_true, y_predicted)

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
    fs = FeatureSelector("None")
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
    model = clf.fit(X=x_train, y=y_train)

    # --------------------------------------------------------------------------------------------------------------
    # Evaluation
    best_model = model
    y_predict_train = best_model.predict(x_train)
    y_predict_validation = best_model.predict(x_validation)

    evaluation_metrics(y_train, y_predict_train, "Train")
    evaluation_metrics(y_validation, y_predict_validation, "Validation")

    if False:
        # train_data_y = tf.keras.utils.to_categorical(train_data_y, 3)
        best_model.fit(pd.DataFrame(x_train).append(pd.DataFrame(x_validation), ignore_index=True),
                       pd.DataFrame(y_train).append(pd.DataFrame(y_validation), ignore_index=True))
        output_submission(best_model, x_test, id_test)
