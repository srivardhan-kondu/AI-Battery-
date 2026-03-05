import os
from datetime import timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    # Flask
    SECRET_KEY = os.getenv("SECRET_KEY", "battery-recycling-super-secret-key-2024")
    DEBUG = os.getenv("DEBUG", "True") == "True"

    # Database
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{os.path.join(BASE_DIR, 'battery_recycling.db')}"
    )
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

    # Tesseract
    TESSERACT_CMD = os.getenv("TESSERACT_CMD", "/opt/homebrew/bin/tesseract")
