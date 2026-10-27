"""Legacy bottom detector from echoedge2.

The detector in this module intentionally mirrors ``echoedge2/lib/find_bottom.py``.
The pipeline adapter is separate so the detector remains a NumPy-array algorithm.
"""

import numpy as np
import pandas as pd


def moving_average(data, window_size):
    series = pd.Series(data)
    moving_averages = series.rolling(window=window_size, center=True, min_periods=1).mean()
    return moving_averages.tolist()


def interpolate_nan(lst, depth_if_all_nan):
    arr = np.array(lst)
    nan_indices = np.isnan(arr)
    non_nan_indices = np.arange(len(arr))[~nan_indices]

    if len(non_nan_indices) == 0:
        return [depth_if_all_nan] * len(arr)

    arr[nan_indices] = np.interp(
        np.arange(len(arr))[nan_indices],
        non_nan_indices,
        arr[non_nan_indices],
    )
    return arr.tolist()


def replace_outliers_with_nan(data):
    def detect_outliers(data, threshold=3):
        mean = np.mean(data)
        std_dev = np.std(data)
        z_scores = [(x - mean) / std_dev for x in data]
        return np.abs(z_scores) > threshold

    outliers_mask = detect_outliers(data)
    data = np.asarray(data, dtype=float)
    data[outliers_mask] = np.nan
    return data


def get_beam_dead_zone(echodata):
    echodata[np.isnan(echodata)] = 0
    row_sums = np.mean(echodata, axis=1).tolist()
    for i, row in enumerate(row_sums):
        if row == row:
            if row < (-50):
                return i


def find_bottom(echodata, window_size, hardness_thresh):
    echodata_original = echodata.copy()
    dead_zone = get_beam_dead_zone(echodata)
    echodata = echodata[dead_zone:, :]

    depth = np.argmax(echodata, axis=0)
    hardness = echodata[depth, np.arange(echodata.shape[1])]

    weak_ping_mask = np.isnan(np.where(hardness < hardness_thresh, np.nan, depth))
    depth = replace_outliers_with_nan(
        np.where(hardness < hardness_thresh, echodata.shape[0], depth)
    )
    depth[weak_ping_mask] = np.nan

    non_nan_depth = depth[~np.isnan(depth)]
    if len(non_nan_depth) == 0:
        depth_roughness = 0
    else:
        depth_roughness = np.round(np.median(np.abs(np.diff(non_nan_depth))), 2)

    depth = interpolate_nan(depth, echodata.shape[0])
    depth = moving_average(depth, window_size)

    depth = [int(item + dead_zone) for item in depth]
    depth = [item - 30 if item != echodata_original.shape[0] else item for item in depth]

    for i in range(0, len(depth)):
        echodata_original[depth[i] :, i] = 0

    return depth, hardness, depth_roughness, echodata_original
