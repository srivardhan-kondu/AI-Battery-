import os
import platform
import shutil
from datetime import timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_tesseract():
    """Cross-platform Tesseract binary detection."""
    env = os.getenv("TESSERACT_CMD")
    if env and os.path.exists(env):
        return env
    found = shutil.which("tesseract")
    if found:
        return found
    candidates = {
        "Darwin": ["/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract"],
        "Linux":  ["/usr/bin/tesseract", "/usr/local/bin/tesseract"],
        "Windows": [
            os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"),
                         "Tesseract-OCR", "tesseract.exe"),
            os.path.join(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"),
                         "Tesseract-OCR", "tesseract.exe"),
        ],
    }
    for p in candidates.get(platform.system(), []):
        if os.path.exists(p):
            return p
    return "tesseract"  # hope it's on PATH


class Config:
    # Flask
    SECRET_KEY = os.getenv("SECRET_KEY", "battery-recycling-super-secret-key-2024")
    DEBUG = os.getenv("DEBUG", "True") == "True"

    # Database (normalize backslashes for SQLite URI on Windows)
    _db_path = os.path.join(BASE_DIR, "battery_recycling.db").replace("\\", "/")
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", f"sqlite:///{_db_path}")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # JWT
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "jwt-battery-secret-2024")
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)
    JWT_BLOCKLIST_ENABLED = True
    JWT_BLOCKLIST_TOKEN_CHECKS = ["access"]

    # File Upload
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    ALLOWED_EXTENSIONS = {"jpeg", "jpg", "png", "bmp"}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

    # ML Models
    MODEL_PATH = os.path.join(BASE_DIR, "models", "battery_detector.pth")
    DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
    MATERIALS_DB = os.path.join(BASE_DIR, "app", "materials_db.json")

    # Tesseract (auto-detected for macOS / Linux / Windows)
    TESSERACT_CMD = _find_tesseract()
