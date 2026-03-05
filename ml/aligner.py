"""
Stage 2: Text Visibility and Alignment Check (FR7)
Uses OpenCV to detect whether text is visible and properly oriented.
"""
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def check_text_alignment(image_path: str) -> dict:
    """
    Checks:
      1. Whether text-like regions are visible (contrast + edge analysis)
      2. Whether text is properly horizontally aligned (Hough line transform)

    Returns:
        {
          "text_visible": bool,
          "aligned": bool,
          "angle": float,  # deviation from horizontal in degrees
          "contrast_score": float,
          "edge_density": float
        }
    """
    if not CV2_AVAILABLE:
        return {
            "text_visible": True,
            "aligned": True,
            "angle": 0.0,
            "method": "fallback",
            "note": "OpenCV not available"
        }

    img = cv2.imread(image_path)
    if img is None:
        return {
            "text_visible": False,
            "aligned": False,
            "angle": 0.0,
            "error": "Cannot read image"
        }

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── 1. Contrast / Brightness check ──────────────────────────────────────
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    # Good contrast: std > 30, not too dark/bright
    contrast_score = round(min(std_val / 128.0, 1.0), 4)
    good_contrast = std_val > 25 and 20 < mean_val < 235

    # ── 2. Edge density (text regions have high edge density) ───────────────
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 40, 120)
    edge_density = float(np.sum(edges > 0)) / edges.size
    has_text_edges = edge_density > 0.04

    text_visible = good_contrast and has_text_edges

    # ── 3. Hough line alignment check ───────────────────────────────────────
    # Look for dominant horizontal lines (text baselines)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=30,
        maxLineGap=10
    )

    angle = 0.0
    aligned = True

    if lines is not None and len(lines) > 0:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(a)

        if angles:
            # Median angle relative to horizontal
            median_angle = float(np.median(angles))
            # Normalize to [-90, 90]
            if median_angle > 90:
                median_angle -= 180
            elif median_angle < -90:
                median_angle += 180
            angle = round(median_angle, 2)
            # Consider aligned if within ±15 degrees of horizontal
            aligned = abs(angle) <= 15.0

    return {
        "text_visible": text_visible,
        "aligned": aligned,
        "angle": angle,
        "contrast_score": contrast_score,
        "edge_density": round(edge_density, 5),
        "mean_brightness": round(mean_val, 2),
        "method": "opencv_hough"
    }
