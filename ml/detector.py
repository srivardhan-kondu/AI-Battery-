"""
Stage 1: Battery Detection (FR6)
Uses a fine-tuned MobileNetV2 classifier.
Falls back to heuristic color/shape analysis if model weights are absent.
"""
import os
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    import torchvision.models as models
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def _build_model(num_classes=2):
    """Return MobileNetV2 fine-tuned for binary battery classification."""
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def _heuristic_detect(image_path: str) -> dict:
    """
    Heuristic battery detection via OpenCV:
    Looks for cylindrical/rectangular shapes, metallic colors.
    """
    if not CV2_AVAILABLE:
        return {
            "battery_detected": True,
            "confidence": 0.75,
            "method": "fallback_default",
            "note": "OpenCV not available; assuming battery present"
        }

    img = cv2.imread(image_path)
    if img is None:
        return {"battery_detected": False, "confidence": 0.0, "method": "heuristic", "error": "Cannot read image"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect circles (cylindrical batteries AA/AAA)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=50, param1=100, param2=30,
        minRadius=20, maxRadius=300
    )

    # Edge density check — batteries tend to have strong edges
    edges = cv2.Canny(blurred, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # Rectangular contour check
    _, thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rect_score = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h if h > 0 else 0
        # Batteries are roughly 1:3 to 1:6 or 1:1 to 2:1
        if 0.15 < aspect < 6.5:
            rect_score += 1

    circle_detected = circles is not None
    confidence = 0.0

    if circle_detected:
        confidence += 0.45
    if rect_score > 0:
        confidence += 0.30
    if edge_density > 0.05:
        confidence += 0.25

    confidence = min(confidence, 0.92)
    battery_detected = confidence > 0.40

    return {
        "battery_detected": battery_detected,
        "confidence": round(confidence, 3),
        "method": "heuristic",
        "circles_found": circle_detected,
        "rectangular_shapes": rect_score,
        "edge_density": round(float(edge_density), 4)
    }


def detect_battery(image_path: str, model_path: str = None) -> dict:
    """
    Main detection function.
    1. Try CNN model if weights exist.
    2. Fall back to heuristic detection.
    """
    if not os.path.exists(image_path):
        return {"battery_detected": False, "confidence": 0.0, "error": "Image not found"}

    # ── Try trained model ────────────────────────────────────────────────────
    if TORCH_AVAILABLE and model_path and os.path.exists(model_path):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = _build_model(num_classes=2)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])

            img = Image.open(image_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                battery_class = torch.argmax(probs).item()
                confidence = probs[battery_class].item()

            return {
                "battery_detected": battery_class == 1,
                "confidence": round(confidence, 4),
                "method": "cnn_mobilenetv2",
                "model_path": model_path
            }
        except Exception as e:
            pass  # Fall through to heuristic

    # ── Heuristic fallback ───────────────────────────────────────────────────
    return _heuristic_detect(image_path)
