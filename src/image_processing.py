import cv2
import numpy as np
import requests
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# -------------------------
# SESSION ROBUSTA CON RETRY
# -------------------------

def create_session():
    session = requests.Session()

    retry = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[403, 429, 500, 502, 503, 504],
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.zara.com/",
    })

    return session


SESSION = create_session()


# -------------------------
# DESCARGA CONTROLADA
# -------------------------

def download_image(url, timeout=20):
    r = SESSION.get(url, timeout=timeout)
    r.raise_for_status()

    arr = np.frombuffer(r.content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Image decode failed")

    # Pequeño sleep para no saturar CDN
    time.sleep(0.03)

    return img


# -------------------------
# DETECCIÓN OBJETO
# -------------------------

def smart_crop(img):
    """
    Detecta bounding box del objeto usando bordes.
    Expande ligeramente el contorno.
    Nunca sale de los límites.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)

    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Expandimos 5% del tamaño
    pad_w = int(w * 0.05)
    pad_h = int(h * 0.05)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img.shape[1], x + w + pad_w)
    y2 = min(img.shape[0], y + h + pad_h)

    return img[y1:y2, x1:x2]


# -------------------------
# RESIZE CORRECTO SIN NEGRO
# -------------------------

def resize_with_white_padding(img, target_size=224):
    """
    Mantiene proporción.
    Centra el objeto.
    Rellena con blanco (no negro).
    """

    h, w = img.shape[:2]

    scale = target_size / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)

    resized = cv2.resize(img, (new_w, new_h))

    # Fondo blanco
    canvas = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255

    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2

    canvas[y_offset:y_offset + new_h,
           x_offset:x_offset + new_w] = resized

    return canvas


# -------------------------
# PIPELINE FINAL
# -------------------------

def preprocess_image(img, img_size=224):
    img = smart_crop(img)
    img = resize_with_white_padding(img, img_size)
    return img