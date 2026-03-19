"""
FR10–FR11: EDA — Battery Material Recovery Percentages
Generates recyclable elements from specific batteries and shows recovery
percentages, processes, hazard levels, and economic value indicators.
"""
import json
import os
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required

eda_bp = Blueprint("eda", __name__)


def load_materials_db(path):
    with open(path, "r") as f:
        return json.load(f)


def _compute_summary(elements):
    """Compute aggregate recovery summary statistics."""
    if not elements:
        return {}

    all_avgs = []
    highest = {"element": None, "avg": 0}
    lowest = {"element": None, "avg": 100}
    hazard_elements = []
    high_value_elements = []

    for name, info in elements.items():
        avg = round((info["recovery_min"] + info["recovery_max"]) / 2, 1)
        all_avgs.append(avg)

        if avg > highest["avg"]:
            highest = {"element": name, "avg": avg, "symbol": info["symbol"]}
        if avg < lowest["avg"]:
            lowest = {"element": name, "avg": avg, "symbol": info["symbol"]}

        hazard = info.get("hazard_level", "low")
        if hazard in ("high", "critical"):
            hazard_elements.append({
                "element": name,
                "symbol": info["symbol"],
                "hazard_level": hazard
            })

        if info.get("economic_value") == "high":
            high_value_elements.append({
                "element": name,
                "symbol": info["symbol"],
                "recovery_avg": avg
            })

    overall_avg = round(sum(all_avgs) / len(all_avgs), 1) if all_avgs else 0
    total_potential = round(sum(all_avgs), 1)

    return {
        "overall_avg_recovery": overall_avg,
        "total_recovery_potential": total_potential,
        "element_count": len(elements),
        "highest_recovery": highest,
        "lowest_recovery": lowest,
        "hazard_elements": hazard_elements,
        "high_value_elements": high_value_elements,
        "has_hazards": len(hazard_elements) > 0
    }


@eda_bp.route("/recover", methods=["POST"])
@jwt_required()
def get_recovery_data():
    """
    FR10: Given a battery chemistry type (and optionally voltage),
    return recoverable elements with min/max/avg recovery percentages,
    recovery processes, hazard levels, and economic value.
    FR11: Process different secondary battery types accordingly.
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
        "colors": [],
        "processes": [],
        "economic_values": [],
        "hazard_levels": []
    }

    # Color palette — hazard/value-aware
    color_map = {
        "critical": "#dc2626",
        "high": "#ea580c",
        "moderate": "#f59e0b",
        "low": "#2563eb"
    }
    fallback_colors = [
        "#2563eb", "#7c3aed", "#059669", "#dc2626",
        "#d97706", "#0891b2", "#6366f1", "#16a34a"
    ]

    for i, (element, info) in enumerate(elements.items()):
        hazard = info.get("hazard_level", "low")
        chart_data["labels"].append(element)
        chart_data["symbols"].append(info["symbol"])
        chart_data["min_values"].append(info["recovery_min"])
        chart_data["max_values"].append(info["recovery_max"])
        avg = round((info["recovery_min"] + info["recovery_max"]) / 2, 1)
        chart_data["avg_values"].append(avg)
        chart_data["notes"].append(info["notes"])
        chart_data["processes"].append(info.get("process", ""))
        chart_data["economic_values"].append(info.get("economic_value", "low"))
        chart_data["hazard_levels"].append(hazard)

        if hazard in ("critical", "high"):
            chart_data["colors"].append(color_map[hazard])
        else:
            chart_data["colors"].append(fallback_colors[i % len(fallback_colors)])

    summary = _compute_summary(elements)

    return jsonify({
        "chemistry": matched_key,
        "full_name": battery_data.get("full_name", matched_key),
        "voltage": voltage,
        "typical_voltage": battery_data.get("typical_voltage", "N/A"),
        "common_brands": battery_data.get("common_brands", []),
        "description": battery_data.get("description", ""),
        "common_applications": battery_data.get("common_applications", []),
        "recycling_process": battery_data.get("recycling_process", ""),
        "environmental_notes": battery_data.get("environmental_notes", ""),
        "element_count": len(elements),
        "chart_data": chart_data,
        "raw_elements": elements,
        "summary": summary
    }), 200


@eda_bp.route("/chemistries", methods=["GET"])
def list_chemistries():
    """Return all available battery chemistries with metadata."""
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
                "element_count": len(val.get("recyclable_elements", {})),
                "description": val.get("description", ""),
                "common_applications": val.get("common_applications", []),
                "has_hazards": any(
                    el.get("hazard_level") in ("high", "critical")
                    for el in val.get("recyclable_elements", {}).values()
                )
            }
            for key, val in materials_db.items()
        ]
    }), 200
