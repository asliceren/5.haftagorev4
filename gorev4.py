# -*- coding: utf-8 -*-
import cv2
import numpy as np

def manuel_bgr_to_rgb(img_bgr):
    img_rgb = np.empty_like(img_bgr)
    img_rgb[:, :, 0] = img_bgr[:, :, 2]
    img_rgb[:, :, 1] = img_bgr[:, :, 1]
    img_rgb[:, :, 2] = img_bgr[:, :, 0]
    return img_rgb

def manuel_bgr_to_gray(img_bgr):
    b = img_bgr[:, :, 0].astype(np.float32)
    g = img_bgr[:, :, 1].astype(np.float32)
    r = img_bgr[:, :, 2].astype(np.float32)
    gri = 0.114 * b + 0.587 * g + 0.299 * r
    return np.round(gri).astype(np.uint8)

DONUSUM_TABLOSU = {
    "BGR2RGB": {
        "manuel_fn": manuel_bgr_to_rgb,
        "cv_kod": cv2.COLOR_BGR2RGB
    },
    "BGR2GRAY": {
        "manuel_fn": manuel_bgr_to_gray,
        "cv_kod": cv2.COLOR_BGR2GRAY
    }
}

def renk_uzayi_donustur(gorsel, donusum_tipi):
    if donusum_tipi not in DONUSUM_TABLOSU:
        raise ValueError(f"Desteklenmeyen tip: {donusum_tipi}")
    
    secilen = DONUSUM_TABLOSU[donusum_tipi]
    manuel_sonuc = secilen["manuel_fn"](gorsel)
    opencv_sonuc = cv2.cvtColor(gorsel, secilen["cv_kod"])
    
    mutlak_fark = cv2.absdiff(opencv_sonuc, manuel_sonuc)
    toplam_hata = np.sum(mutlak_fark)
    ortalama_hata = np.mean(mutlak_fark)
    
    return manuel_sonuc, opencv_sonuc, toplam_hata, ortalama_hata

resim = cv2.imread("renklifoto.jpg")
if resim is None:
    print("Hata: 'renklifoto.jpg' bulunamadi!")
    exit()

for donusum in ["BGR2RGB", "BGR2GRAY"]:
    manuel_img, cv_img, sad, mae = renk_uzayi_donustur(resim, donusum)
    print(f"Donusum: {donusum}")
    print(f"Toplam Hata (SAD): {sad}")
    print(f"Ortalama Hata: {mae:.5f}")
    print("-" * 30)

fark_rgb = cv2.absdiff(cv2.cvtColor(resim, cv2.COLOR_BGR2RGB), manuel_bgr_to_rgb(resim))
cv2.imwrite("fark_bgr2rgb.png", fark_rgb)
