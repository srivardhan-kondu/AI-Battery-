"""
FR10–FR11: EDA — Battery Material Recovery Percentages
"""
import json
import os
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required

eda_bp = Blueprint("eda", __name__)


def load_materials_db(path):
    with open(path, "r") as f:
        return json.load(f)


@eda_bp.route("/recover", methods=["POST"])
@jwt_required()
def get_recovery_data():
    """
    FR10: Given a battery chemistry type (and optionally voltage),
    return recoverable elements with min/max/avg recovery percentages.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    chemistry = data.get("chemistry", "").strip()
    voltage = data.get("voltage", "Unknown")

    if not chemistry:
        return jsonify({"error": "chemistry field is required"}), 400

    db_path = current_app.config["MATERIALS_DB"]
    if not os.path.exists(db_path):
        return jsonify({"error": "Materials database not found"}), 500

    materials_db = load_materials_db(db_path)

    # Try direct match, then partial match
    matched_key = None
    for key in materials_db:
        if key.lower() == chemistry.lower():
            matched_key = key
            break

    if not matched_key:
        for key in materials_db:
            if chemistry.lower() in key.lower() or key.lower() in chemistry.lower():
                matched_key = key
                break

    if not matched_key:
        available = list(materials_db.keys())
        return jsonify({
            "error": f"Chemistry '{chemistry}' not found in database",
            "available_chemistries": available
        }), 404

    battery_data = materials_db[matched_key]
    elements = battery_data["recyclable_elements"]

    # Build chart-ready response
    chart_data = {
        "labels": [],
        "symbols": [],
        "min_values": [],
        "max_values": [],
        "avg_values": [],
        "notes": [],
        "colors": []
    }

    # Color palette for chart bars
    colors = [
        "#00d4ff", "#7b2ff7", "#00ff88", "#ff6b6b",
        "#ffd93d", "#ff8c42", "#a8dadc", "#6bcb77"
    ]

    for i, (element, info) in enumerate(elements.items()):
        chart_data["labels"].append(element)
        chart_data["symbols"].append(info["symbol"])
        chart_data["min_values"].append(info["recovery_min"])
        chart_data["max_values"].append(info["recovery_max"])
        avg = round((info["recovery_min"] + info["recovery_max"]) / 2, 1)
        chart_data["avg_values"].append(avg)
        chart_data["notes"].append(info["notes"])
        chart_data["colors"].append(colors[i % len(colors)])

    return jsonify({
        "chemistry": matched_key,
        "full_name": battery_data.get("full_name", matched_key),
        "voltage": voltage,
        "typical_voltage": battery_data.get("typical_voltage", "N/A"),
        "common_brands": battery_data.get("common_brands", []),
        "element_count": len(elements),
        "chart_data": chart_data,
        "raw_elements": elements
    }), 200


@eda_bp.route("/chemistries", methods=["GET"])
def list_chemistries():
    """Return all available battery chemistries."""
    db_path = current_app.config["MATERIALS_DB"]
    if not os.path.exists(db_path):
        return jsonify({"error": "Materials database not found"}), 500

    materials_db = load_materials_db(db_path)
    return jsonify({
        "chemistries": [
            {
                "key": key,
                "full_name": val.get("full_name", key),
                "typical_voltage": val.get("typical_voltage", "N/A"),
                "element_count": len(val.get("recyclable_elements", {}))
            }
            for key, val in materials_db.items()
        ]
    }), 200
