"""
Stage 3: OCR Extraction — Brand, Chemistry, Voltage (FR8)
Uses Tesseract OCR + regex patterns to extract battery labels.
"""
import re
import os

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ── Known brand lexicon ──────────────────────────────────────────────────────
BATTERY_BRANDS = [
    "LG", "Samsung", "Panasonic", "Sony", "Duracell", "Energizer",
    "Rayovac", "Varta", "BYD", "CATL", "Tesla", "A123", "Winston",
    "Saft", "GP", "Amara Raja", "Amaron", "Bosch", "Delkor", "Exide",
    "Eneloop", "Eveready", "Sanyo", "Toshiba", "Hitachi", "Murata",
    "EVE", "CALB", "Lishen", "BAK", "ATL"
]

# ── Chemistry patterns ───────────────────────────────────────────────────────
CHEMISTRY_PATTERNS = {
    "Li-ion": [
        r"li[\s\-]?ion", r"lithium[\s\-]?ion", r"li[\s\-]?ion[\s\-]?battery",
        r"ICR", r"INR", r"IMR", r"NCR"
    ],
    "LiFePO4": [
        r"lifepo4", r"lfp", r"lithium[\s\-]?iron[\s\-]?phosphate",
        r"life[\s\-]?po", r"li[\s\-]?fe"
    ],
    "NiMH": [
        r"ni[\s\-]?mh", r"nickel[\s\-]?metal[\s\-]?hydride",
        r"nimh", r"ni\-mh"
    ],
    "NiCd": [
        r"ni[\s\-]?cd", r"nickel[\s\-]?cadmium",
        r"nicd", r"ni\-cd", r"cadmium"
    ],
    "Lead-Acid": [
        r"lead[\s\-]?acid", r"vrla", r"sla", r"agm",
        r"gel[\s\-]?cell", r"flooded[\s\-]?lead"
    ],
    "Alkaline": [
        r"alkaline", r"zinc[\s\-]?mno2", r"zinc[\s\-]?manganese",
        r"lr\d{2}", r"am\d"
    ]
}

# ── Voltage pattern ──────────────────────────────────────────────────────────
VOLTAGE_PATTERN = re.compile(
    r"(\d{1,3}(?:\.\d{1,2})?)\s*(?:V|v|volts?|VOLT)",
    re.IGNORECASE
)


def _preprocess_for_ocr(image_path: str) -> "Image":
    """
    Enhance image contrast and sharpness for better OCR accuracy.
    """
    if not PIL_AVAILABLE:
        return None

    img = Image.open(image_path).convert("RGB")

    # Resize if small
    w, h = img.size
    if w < 400 or h < 400:
        scale = max(400 / w, 400 / h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Sharpen
    img = img.filter(ImageFilter.SHARPEN)
    # Increase contrast
    img = ImageEnhance.Contrast(img).enhance(2.0)
    # Increase sharpness
    img = ImageEnhance.Sharpness(img).enhance(2.0)

    return img


def _extract_brand(text: str) -> str:
    text_upper = text.upper()
    for brand in sorted(BATTERY_BRANDS, key=len, reverse=True):
        if brand.upper() in text_upper:
            return brand
    return "Unknown"


def _extract_chemistry(text: str) -> str:
    text_lower = text.lower()
    for chemistry, patterns in CHEMISTRY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return chemistry
    return "Unknown"


def _extract_voltage(text: str) -> str:
    matches = VOLTAGE_PATTERN.findall(text)
    if matches:
        # Return the most prominent voltage (usually the largest or most common)
        voltages = [float(v) for v in matches]
        # Filter out unrealistic values
        valid = [v for v in voltages if 0.5 <= v <= 500]
        if valid:
            return f"{sorted(valid)[-1]}V"
    return "Unknown"


def extract_battery_info(image_path: str) -> dict:
    """
    Run Tesseract OCR on the image and extract battery metadata.
    Falls back to empty extraction gracefully.
    """
    if not os.path.exists(image_path):
        return {"brand": "Unknown", "chemistry": "Unknown", "voltage": "Unknown",
                "error": "Image not found", "raw_text": ""}

    raw_text = ""

    if TESSERACT_AVAILABLE:
        try:
            # Set tesseract path if needed
            tesseract_paths = [
                "/opt/homebrew/bin/tesseract",    # Apple Silicon Mac
                "/usr/local/bin/tesseract",       # Intel Mac
                "/usr/bin/tesseract",             # Linux
            ]
            for path in tesseract_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break

            # Try multiple PSM modes for better results
            img = _preprocess_for_ocr(image_path)
            configs = [
                "--psm 6 --oem 3",   # Uniform block of text
                "--psm 11 --oem 3",  # Sparse text
                "--psm 3 --oem 3",   # Auto page segmentation
            ]

            texts = []
            for cfg in configs:
                try:
                    t = pytesseract.image_to_string(img or image_path, config=cfg)
                    texts.append(t)
                except Exception:
                    pass

            raw_text = "\n".join(texts)

        except Exception as e:
            raw_text = f"OCR_ERROR: {str(e)}"
    else:
        raw_text = "TESSERACT_NOT_INSTALLED"

    brand = _extract_brand(raw_text)
    chemistry = _extract_chemistry(raw_text)
    voltage = _extract_voltage(raw_text)

    return {
        "brand": brand,
        "chemistry": chemistry,
        "voltage": voltage,
        "raw_text": raw_text[:500],  # Limit for response size
        "tesseract_available": TESSERACT_AVAILABLE
    }
