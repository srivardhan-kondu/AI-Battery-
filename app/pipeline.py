"""
FR9: Unified Python Pipeline — Battery Detection + Text Alignment + OCR
"""
import os
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required
from werkzeug.utils import secure_filename
import uuid

pipeline_bp = Blueprint("pipeline", __name__)


def allowed_file(filename, allowed_extensions):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


@pipeline_bp.route("/analyze", methods=["POST"])
@jwt_required()
def analyze():
    """
    Full 3-stage pipeline:
      Stage 1 → Battery Detection
      Stage 2 → Text Visibility & Alignment Check
      Stage 3 → OCR: Brand / Chemistry / Voltage
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    allowed = current_app.config["ALLOWED_EXTENSIONS"]
    if not allowed_file(file.filename, allowed):
        return jsonify({"error": "Invalid file format. Use JPEG, PNG, or BMP"}), 415

    # Save file temporarily
    ext = file.filename.rsplit(".", 1)[1].lower()
    tmp_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(current_app.config["UPLOAD_FOLDER"], secure_filename(tmp_name))
    file.save(save_path)

    results = {
        "filename": tmp_name,
        "stage_1_detection": {},
        "stage_2_alignment": {},
        "stage_3_ocr": {},
        "summary": {}
    }

    try:
        # ── Stage 1: Battery Detection ──────────────────────────────────────
        from ml.detector import detect_battery
        detection = detect_battery(save_path, current_app.config["MODEL_PATH"])
        results["stage_1_detection"] = detection

        if not detection.get("battery_detected"):
            results["summary"] = {
                "battery_detected": False,
                "message": "No battery found in the image. Please upload a clear battery image."
            }
            return jsonify(results), 200

        # ── Stage 2: Text Alignment Check ───────────────────────────────────
        from ml.aligner import check_text_alignment
        alignment = check_text_alignment(save_path)
        results["stage_2_alignment"] = alignment

        # ── Stage 3: OCR Extraction ─────────────────────────────────────────
        from ml.ocr_extractor import extract_battery_info
        ocr_result = extract_battery_info(save_path)
        results["stage_3_ocr"] = ocr_result

        # ── Summary ─────────────────────────────────────────────────────────
        results["summary"] = {
            "battery_detected": True,
            "text_visible": alignment.get("text_visible", False),
            "text_aligned": alignment.get("aligned", False),
            "brand": ocr_result.get("brand", "Unknown"),
            "chemistry": ocr_result.get("chemistry", "Unknown"),
            "voltage": ocr_result.get("voltage", "Unknown"),
            "confidence": detection.get("confidence", 0.0)
        }

    except Exception as e:
        return jsonify({"error": f"Pipeline error: {str(e)}"}), 500

    return jsonify(results), 200
