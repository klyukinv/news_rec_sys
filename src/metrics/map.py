import numpy as np
import pandas as pd


def ap_score(actual, predicted, k=20):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    actual : set
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    if not actual:
        return 0.0

    score = 0.0
    num_hits = 0.0

    for i, pred_item in enumerate(predicted):
        if pred_item in actual and pred_item not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)


def map_score(actual, predicted, k=20):
    """
    Computes the mean average precision at k.
    This function computes the mean average precision at k between two dataframes
    Parameters
    ----------
    actual : pd.DataFrame
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : pd.DataFrame
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    actual = actual[actual.rating == 1]

    return np.mean([
        ap_score(set(actual[(actual.userId == userId)].itemId), list(predicted[predicted.userId == userId].itemId), k)
        for userId in actual.userId.unique()
    ])
