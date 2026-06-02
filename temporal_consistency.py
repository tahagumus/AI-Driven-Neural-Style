"""
temporal_consistency.py - Optical flow ile frame'ler arası titreme engelleme
Ruder et al. (2016) - https://arxiv.org/abs/1604.08610
"""

import cv2
import numpy as np


def optical_flow(prev_gray, curr_gray):
    return cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )


def warp(frame, flow):
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                          np.arange(h, dtype=np.float32))
    return cv2.remap(frame,
                     (gx + flow[:,:,0]).astype(np.float32),
                     (gy + flow[:,:,1]).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


class TemporalSmoother:
    def __init__(self, alpha=0.7):
        self.alpha = alpha        # 0=sabit önceki, 1=tamamen yeni
        self._prev_gray = None
        self._prev_styled = None

    def apply(self, frame, styled):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray, self._prev_styled = gray, styled.copy()
            return styled

        flow    = optical_flow(self._prev_gray, gray)
        warped  = warp(self._prev_styled, flow)
        result  = (self.alpha * styled + (1 - self.alpha) * warped).astype(np.uint8)

        self._prev_gray, self._prev_styled = gray, result.copy()
        return result

    def reset(self):
        self._prev_gray = self._prev_styled = None
