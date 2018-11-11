import keras.backend as K
import numpy as np


def precision_tensor(y_true, y_pred, threshold=0.5):
    """
    Reimplemented removed Keras metric.
    Sourced from https://github.com/keras-team/keras/commit/a56b1a55182acf061b1eb2e2c86b48193a0e88f7

    Precision metric.
     Only computes a batch-wise average of precision.
     Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    prec = true_positives / (predicted_positives + K.epsilon())
    return prec


def precision(y_true, y_pred):
    # true_positives = np.sum(np.around(np.clip(y_true * y_pred, 0, 1)))
    # predicted_positives = np.sum(np.around(np.clip(y_pred, 0, 1)))
    # prec = true_positives / predicted_positives
    TP = np.count_nonzero(np.logical_and(y_pred.flatten() >= 0.5, y_true.flatten() == 1))
    FP = np.count_nonzero(np.logical_and(y_pred.flatten() >= 0.5, y_true.flatten() == 0))
    if TP + FP == 0:
        return 0.
    else:
        return TP/(TP+FP)


def recall_tensor(y_true, y_pred):
    """
    Reimplemented removed Keras metric
    Sourced from https://github.com/keras-team/keras/commit/a56b1a55182acf061b1eb2e2c86b48193a0e88f7

    Recall metric.
     Only computes a batch-wise average of recall.
     Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    rec = true_positives / (possible_positives + K.epsilon())
    return rec


def recall(y_true, y_pred):
    # true_positives = np.sum(np.around(np.clip(y_true * y_pred, 0, 1)))
    # possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    # rec = true_positives / possible_positives
    TP = np.count_nonzero(np.logical_and(y_pred.flatten() >= 0.5, y_true.flatten() == 1))
    FN = np.count_nonzero(np.logical_and(y_pred.flatten() < 0.5, y_true.flatten() == 1))
    if TP + FN == 0:
        return 0.
    else:
        return TP/(TP + FN)


def fbeta_score_tensor(y_true, y_pred, beta=1):
    """
    Reimplemented removed Keras metric.
    Sourced from https://github.com/keras-team/keras/commit/a56b1a55182acf061b1eb2e2c86b48193a0e88f7

    Computes the F score.
     The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
     This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
     With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
    #  If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0
    p = precision_tensor(y_true, y_pred)
    r = recall_tensor(y_true, y_pred)
    bb = beta ** 2
    score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return score


def fbeta_score(y_true, y_pred, beta=1):
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
    #  If there are no true positives, fix the F score at 0 like sklearn.
    if np.sum(np.around(np.clip(y_true, 0, 1))) == 0:
        return 0
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p + r == 0:
        return 0.
    bb = beta ** 2
    score = (1 + bb) * (p * r) / (bb * p + r)
    return score
