import cv2
import numpy as np

def brain_mri_quick_check(img):
    """
    Strict rule-based brain MRI validation.
    Hand MRIs and other images are rejected.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -------------------------
    # 1. Edge Density (HARD BLOCK)
    # -------------------------
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # Hand MRIs usually > 0.18
    if edge_density > 0.14:
        return False

    # -------------------------
    # 2. Contour Area Ratio
    # -------------------------
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return False

    largest = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(largest) / (img.shape[0] * img.shape[1])

    if not (0.30 < area_ratio < 0.70):
        return False

    # -------------------------
    # 3. Aspect Ratio Constraint
    # -------------------------
    h, w = gray.shape
    aspect_ratio = w / h

    # Brain MRIs are near-square
    if not (0.85 < aspect_ratio < 1.15):
        return False

    # -------------------------
    # 4. Center Brightness Bias
    # -------------------------
    center = gray[h//4:3*h//4, w//4:3*w//4]
    outer = gray.copy()
    outer[h//4:3*h//4, w//4:3*w//4] = 0

    if center.mean() <= outer.mean():
        return False

    return True
