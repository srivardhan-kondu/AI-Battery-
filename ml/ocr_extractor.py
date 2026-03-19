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

# ── Brand / model‑number → chemistry inference ──────────────────────────────
# When OCR picks up a model number or brand but no explicit chemistry text,
# we use these patterns to infer the chemistry.
MODEL_CHEMISTRY_PATTERNS = {
    "Li-ion": [
        r"NCR\d{5}",        # Panasonic NCR18650B, NCR18650GA …
        r"INR\d{5}",        # Samsung INR18650-25R, LG INR …
        r"ICR\d{5}",        # ICR18650 cells
        r"IMR\d{5}",        # IMR high-drain cells
        r"18650", r"21700", r"26650", r"14500", r"16340",  # common Li-ion form factors
        r"MJ1", r"HG2", r"VTC[456]", r"25R", r"30Q", r"35E",  # popular cell models
        r"NCR\d{2}", r"CGR\d{5}",
    ],
    "LiFePO4": [
        r"IFR\d{5}",        # IFR form factor prefix
        r"ANR\d{5}",        # A123 cells
        r"32650",           # common LFP form factor
        r"LFP",
    ],
    "NiMH": [
        r"BK-\d+", r"eneloop", r"HR\d{1,2}",  # Panasonic eneloop, HR6 etc.
    ],
    "NiCd": [
        r"KR\d{1,2}",      # NiCd form factor prefix
    ],
    "Alkaline": [
        r"LR\d{1,2}", r"AM\d",  # LR6, AM3 etc.
        r"MN\d{4}",         # Duracell model numbers
    ],
}

# ── Brand → default chemistry (common primary chemistry per brand) ───────────
BRAND_DEFAULT_CHEMISTRY = {
    "Panasonic":  "Li-ion",   # vast majority of Panasonic cells
    "Samsung":    "Li-ion",
    "LG":         "Li-ion",
    "Sony":       "Li-ion",
    "Murata":     "Li-ion",
    "Sanyo":      "Li-ion",
    "Toshiba":    "Li-ion",
    "Hitachi":    "Li-ion",
    "BAK":        "Li-ion",
    "ATL":        "Li-ion",
    "EVE":        "Li-ion",
    "Lishen":     "Li-ion",
    "BYD":        "LiFePO4",
    "CATL":       "LiFePO4",
    "A123":       "LiFePO4",
    "CALB":       "LiFePO4",
    "Duracell":   "Alkaline",
    "Energizer":  "Alkaline",
    "Rayovac":    "Alkaline",
    "Eveready":   "Alkaline",
    "Eneloop":    "NiMH",
    "GP":         "NiMH",
    "Varta":      "NiMH",
}

# ── Voltage → chemistry fallback ─────────────────────────────────────────────
VOLTAGE_CHEMISTRY_MAP = [
    ((3.2, 3.35),  "LiFePO4"),
    ((3.6, 4.3),   "Li-ion"),
    ((1.15, 1.35), "NiMH"),      # also NiCd — NiMH is more common today
    ((1.45, 1.65), "Alkaline"),
    ((2.0, 2.2),   "Lead-Acid"),  # per-cell
    ((6.0, 6.1),   "Lead-Acid"),
    ((12.0, 12.9), "Lead-Acid"),
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


def _preprocess_for_ocr(image_path: str) -> list:
    """
    Enhance image contrast and sharpness for better OCR accuracy.
    Returns a list of preprocessed image variants (original + rotated)
    to handle vertical/rotated text on cylindrical batteries.
    """
    if not PIL_AVAILABLE:
        return []

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

    # Return multiple orientations to handle rotated text on batteries
    variants = [img]
    variants.append(img.rotate(90, expand=True))   # 90° CW text
    variants.append(img.rotate(270, expand=True))   # 90° CCW text

    return variants


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


def _infer_chemistry_from_model(text: str) -> str:
    """Match known model-number patterns to infer chemistry."""
    for chemistry, patterns in MODEL_CHEMISTRY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return chemistry
    return "Unknown"


def _infer_chemistry_from_voltage(voltage_str: str) -> str:
    """Map a detected voltage value to the most likely chemistry."""
    try:
        v = float(re.sub(r"[^\d.]", "", voltage_str))
    except (ValueError, TypeError):
        return "Unknown"
    for (lo, hi), chemistry in VOLTAGE_CHEMISTRY_MAP:
        if lo <= v <= hi:
            return chemistry
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

            # Try multiple PSM modes AND rotations for better results
            img_variants = _preprocess_for_ocr(image_path)
            configs = [
                "--psm 6 --oem 3",   # Uniform block of text
                "--psm 11 --oem 3",  # Sparse text
                "--psm 3 --oem 3",   # Auto page segmentation
            ]

            texts = []
            for img in (img_variants or [image_path]):
                for cfg in configs:
                    try:
                        t = pytesseract.image_to_string(img, config=cfg)
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

    # ── Fallback chain when OCR can't find explicit chemistry text ────────
    if chemistry == "Unknown":
        chemistry = _infer_chemistry_from_model(raw_text)

    if chemistry == "Unknown" and voltage != "Unknown":
        chemistry = _infer_chemistry_from_voltage(voltage)

    if chemistry == "Unknown" and brand != "Unknown":
        chemistry = BRAND_DEFAULT_CHEMISTRY.get(brand, "Unknown")

    return {
        "brand": brand,
        "chemistry": chemistry,
        "voltage": voltage,
        "raw_text": raw_text[:500],  # Limit for response size
        "tesseract_available": TESSERACT_AVAILABLE
    }
