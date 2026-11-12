"""The area where the seed maps mean something, shared by the threshold suggestion (bm_qc) and the Thresholds tab.

Outside it the maps are not data: near the frame border the filters see the edge, and in the dark area outside the
motion-corrected field the filtered movie is rounding noise, so PNR and correlation there are ratios of that noise
and change with the strip heights of the thresholds stage. Kept free of CaImAn so the window can import it.
"""
import cv2
import numpy as np


def valid_mask(maps, border_pix=5, dark_frac=0.25):
    """
    Pixels usable for seeds: not within border_pix of the edge and not in dark regions (outside the lens,
    black corners after cropping/motion correction), where PNR/correlation are meaningless.
    """
    mean = maps.get('mean')
    valid = np.ones(maps['cn'].shape, bool)
    if mean is not None:
        dark = mean < dark_frac * np.median(mean)
        k = 2 * int(2 * maps['gSig']) + 1
        dark = cv2.dilate(dark.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
        valid &= ~dark
    if border_pix > 0:
        valid[:border_pix, :] = valid[-border_pix:, :] = False
        valid[:, :border_pix] = valid[:, -border_pix:] = False
    return valid
