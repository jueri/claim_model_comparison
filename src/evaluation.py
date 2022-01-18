# -*- coding: utf-8 -*-
"""Different functions to plot the predicted results of a model.

Example:
        $ from src.evaluation import roc_plot, confusion_matrix_plot
        $ 
        $ confusion_matrix_plot(y_test, Y_pred, label=[False, True], title=MODEL_NAME+" confusion matrix")


"""
from typing import Any, List

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import seaborn as sn  # type: ignore
from sklearn.metrics import auc, confusion_matrix, roc_curve  # type: ignore


def roc_plot(y_test: pd.Series, y_score: np.ndarray, title: str):
    """Plot a receiver operating characteristic (ROC) curve of the predicted results.
    The closer the curve is to the left top corner, the better.
    The higher the area under the curve (AUC) is the better.

    Resources:
        https://towardsdatascience.com/various-ways-to-evaluate-a-machine-learning-models-performance-230449055f15
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    Args:
        y_test (pd.Series): Actual test results.
        y_score (np.ndarray): Predicted confidence scores for samples.
        title (str): Title of the plot.
    """
    # Compute ROC curve and ROC area for each class
    fpr: dict[Any, Any] = {}
    tpr: dict[Any, Any] = {}
    roc_auc: dict[Any, Any] = {}
    n_classes = len(y_test.unique())
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_score, pos_label=True)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(facecolor="white")
    lw = 2
    plt.plot(
        fpr[0],
        tpr[0],
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc[0],
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def confusion_matrix_plot(
    y_test: pd.Series, y_pred: np.ndarray, label: List[str], title: str
):
    """Plot a confusion matrix for model evaluation.
    The sklearn confusion matrix generator is used. The order of cells is, therefore:
    TN | FP
    FN | TP

    TN = predicted / actual
    Resources: https://towardsdatascience.com/understanding-the-confusion-matrix-from-scikit-learn-c51d88929c79

    Args:
        y_test (pd.Series): Test data samples.
        y_pred (np.ndarray): Predicted classes for test samples.
        label (list[str]): Ordered list of labels.
        title (str): The title for the plot.
    """
    sn.set(rc={"figure.facecolor": "white"})
    cm = pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        index=[i for i in label],
        columns=[i for i in label],
    )
    plt.figure(figsize=(10, 7))
    plt.title(title)

    sn.heatmap(cm, annot=True, fmt="d").set(
        ylabel="Actual Labels", xlabel="Predicted Labels"
    )
