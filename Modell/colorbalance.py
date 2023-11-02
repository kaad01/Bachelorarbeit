import cv2
import math
import numpy as np
import sys
# def simplest_cb(img, percent=1):
#     out_channels = []
#     cumstops = (
#         img.shape[0] * img.shape[1] * percent / 200.0,
#         img.shape[0] * img.shape[1] * (1 - percent / 200.0)
#     )
#     for channel in cv2.split(img):
#         cumhist = np.cumsum(cv2.calcHist([channel], [0], None, [256], (0,256)))
#         low_cut, high_cut = np.searchsorted(cumhist, cumstops)
#         lut = np.concatenate((
#             np.zeros(low_cut),
#             np.around(np.linspace(0, 255, high_cut - low_cut + 1)),
#             255 * np.ones(255 - high_cut)
#         ))
#         out_channels.append(cv2.LUT(channel, lut.astype('uint8')))
#     return cv2.merge(out_channels)



def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()

def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix

def simplest_cb(img, percent):
    assert img.shape[2] == 3
    assert percent > 0 and percent < 100

    half_percent = percent / 200.0

    channels = cv2.split(img)

    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2
        # find the low and high precentile values (based on the input percentile)
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)

        assert len(flat.shape) == 1

        flat = np.sort(flat)

        n_cols = flat.shape[0]

        low_val  = flat[math.floor(n_cols * half_percent)]
        high_val = flat[math.ceil( n_cols * (1.0 - half_percent))]


        # saturate below the low percentile and above the high percentile
        thresholded = apply_threshold(channel, low_val, high_val)
        # scale the channel
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)
