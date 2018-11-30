import pandas
import numpy as np
import os


def make_timeseries_instances(timeseries, window_size, proportional_step):
    """Make input features and prediction targets from a `timeseries` for use
    in machine learning.

    return: A tuple of `(X, y, q)`.  `X` are the inputs to a predictor,
    a 3D ndarray with shape:
    ``(timeseries.shape[0] - window_size,
    window_size,
    timeseries.shape[1] or 1)``

    For each row of `X`, the corresponding row of `y` is the next value in the
    timeseries.  The `q` or query is the last instance, what you would use
    to predict a hypothetical next (unprovided) value in the `timeseries`.

    Parameters
    ----------
    ndarray timeseries: Either a simple vector, or a matrix of shape:
    ``(timestep, series_num)``
    i.e., time is axis 0 (the row) and the series is axis 1 (the column).

    int window_size: The number of samples to use as input prediction
    features (also called the lag or lookback).

    int/float proportional_step is the proportional overlap there
    should be between windows.
    =1 leads to sliding one increment forward
    =0 leads to no overlap at all
    """

    timeseries = np.asarray(timeseries)
    assert 0 < window_size < timeseries.shape[0]

    if proportional_step == 1:
        X = np.atleast_3d(np.array([timeseries[start:start + window_size]
                                    for start in range(0, timeseries.shape[0] -
                                                       window_size)]))

    elif proportional_step == 0:
        X = np.atleast_3d(np.array([timeseries[start:start + window_size]
                                    for start in range(0, timeseries.shape[0] -
                                                       window_size,
                                                       window_size)]))
    else:
        step_size = int(proportional_step * window_size)
        X = np.atleast_3d(np.array([timeseries[start:start + window_size]
                                    for start in range(0, timeseries.shape[0] -
                                                       window_size,
                                                       step_size)]))

    return X


def load_raw_csv_file(directory, participant_id, usecols=[1, 2, 3, 4]):
    """ Load in csv files (including compressed csv files) of the raw time
    series data from CAPTURE-24 or UK biobank

    return: array of x-, y-, z-axis acceleration measured in units of g
    and also activity at each step (4th column, only for CAPTURE-24)

    Parameters
    ----------
    string directory: dir containg files
    string particpant_id: identifier of file to be loaded
    list usecols (optional): which columns to load from file. If being used
    with CAPTURE-24 data the default is correct and it will load activty class
    as well as 4th col.
    """
    df = pandas.read_csv(os.path.join(directory, participant_id),
                         sep=',', usecols=usecols)
    data = df.values
    print('loaded ', participant_id)
    return data


def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    return: an array the same size as X. The result will sum to 1
    along the specified axis.

    Parameters
    ----------
    ndarray X (probably should be floats)
    float theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    int axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
    # finally: divide elementwise
    p = y / ax_sum
    # flatten if X was 1D
    if len(X.shape) == 1:
        p = p.flatten()
    return p
