"""
style_transfer.py - Stil Aktarımı
Renk paleti transferi + doku sentezi (VGG gerektirmez, tamamen çalışır)
"""

import cv2
import numpy as np
from PIL import Image

SIZE = 256


def color_transfer(content, style):
    """
    Reinhard (2001) LAB renk uzayında ortalama/std transferi.
    Stil görüntüsünün renk paletini içeriğe uygular.
    """
    c = cv2.cvtColor(content, cv2.COLOR_BGR2LAB).astype(np.float32)
    s = cv2.cvtColor(style,   cv2.COLOR_BGR2LAB).astype(np.float32)

    for ch in range(3):
        c_mean, c_std = c[:,:,ch].mean(), c[:,:,ch].std()
        s_mean, s_std = s[:,:,ch].mean(), s[:,:,ch].std()
        # Normalize et, stil istatistiklerine taşı
        c[:,:,ch] = (c[:,:,ch] - c_mean) * (s_std / (c_std + 1e-6)) + s_mean

    result = cv2.cvtColor(np.clip(c, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return result


def texture_blend(content, style, alpha=0.35):
    """
    Frekans ayrıştırma: içeriğin yapısı + stilin dokusu.
    Gaussian piramit ile düşük/yüksek frekansları ayır ve karıştır.
    """
    # Düşük frekans (genel şekil) - içerikten
    c_blur = cv2.GaussianBlur(content, (21, 21), 0)
    c_high = cv2.subtract(content, c_blur)  # yüksek frekans (kenarlar)

    # Stilden doku al
    s_resized = cv2.resize(style, (content.shape[1], content.shape[0]))
    s_blur = cv2.GaussianBlur(s_resized, (21, 21), 0)
    s_high = cv2.subtract(s_resized, s_blur)

    # Karıştır: içerik yapısı + stil dokusu
    blended_low  = cv2.addWeighted(c_blur, 1 - alpha, s_blur, alpha, 0)
    blended_high = cv2.addWeighted(c_high, 0.6,       s_high, 0.4,  0)
    return cv2.add(blended_low, blended_high)


def painterly_effect(img):
    """
    Yağlı boya efekti: kenar korumalı düzleştirme + kontur vurgulama.
    """
    # Bilateral filter: kenarları koru, yüzeyleri düzleştir (boya damlası efekti)
    smooth = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    smooth = cv2.bilateralFilter(smooth, d=9, sigmaColor=75, sigmaSpace=75)

    # Kenarlara Van Gogh tarzı kontur ekle
    gray  = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, blockSize=7, C=2)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(smooth, edges)


def stylize(content_img, style_img, blend=0.4, painterly=True):
    """
    Ana stil aktarım pipeline:
      1. Renk paleti transferi
      2. Doku karıştırma
      3. Yağlı boya efekti
    """
    
    c = cv2.resize(content_img, (SIZE, SIZE))
    s = cv2.resize(style_img,   (SIZE, SIZE))

    result = color_transfer(c, s)
    result = texture_blend(result, s, alpha=blend)
    if painterly:
        result = painterly_effect(result)

    return result


# yardımcı kodlar


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Görüntü bulunamadı: {path}")
    return img

def frame_to_array(bgr):
    return cv2.resize(bgr, (SIZE, SIZE))
