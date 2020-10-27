import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import balanced_accuracy_score, multilabel_confusion_matrix, classification_report
from sklearn.metrics import confusion_matrix

from logcreator.logcreator import Logcreator


def plot_individual_cm(y_true, y_predicted):
    cm = multilabel_confusion_matrix(y_true, y_predicted, labels=[0, 1, 2])

    for i in range(cm.shape[0]):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm[i],
                                      display_labels=[i, "rest"])
        disp = disp.plot()
    plt.show()


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
        Logcreator.info("Normalized confusion matrix")
    else:
        Logcreator.info('Confusion matrix, without normalization')

    Logcreator.info(cm)

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
        if isinstance(y, pandas.DataFrame):
            y = y.values
        y = np.argmax(y, axis=1)
    return y


def evaluation_metrics(y_true, y_predicted, text):
    Logcreator.info("\n---------", text, "---------\n")
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
    Logcreator.info(text, 'report')
    Logcreator.info(classification_report(y_true, y_predicted))

    # balanced accuracy score
    score = balanced_accuracy_score(y_true, y_predicted)
    Logcreator.info("bas_score on", text, "split:", score)
