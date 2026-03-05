"""
FR1: User Authentication — Register / Login / Logout
"""
from flask import Blueprint, request, jsonify
from flask_jwt_extended import (
    create_access_token, jwt_required, get_jwt
)
from app import db, bcrypt, BLOCKLISTED_TOKENS
from app.models import User
import re

auth_bp = Blueprint("auth", __name__)


def is_valid_email(email):
    return re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", email)


def is_strong_password(pw):
    """At least 8 chars, 1 uppercase, 1 lowercase, 1 digit."""
    return (
        len(pw) >= 8 and
        any(c.isupper() for c in pw) and
        any(c.islower() for c in pw) and
        any(c.isdigit() for c in pw)
    )


@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    if not is_valid_email(email):
        return jsonify({"error": "Invalid email format"}), 400

    if not is_strong_password(password):
        return jsonify({
            "error": "Password must be at least 8 characters with uppercase, lowercase, and a digit"
        }), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already registered"}), 409

    hashed = bcrypt.generate_password_hash(password).decode("utf-8")
    user = User(email=email, password_hash=hashed)
    db.session.add(user)
    db.session.commit()

    token = create_access_token(identity=str(user.id))
    return jsonify({
        "message": "Registration successful",
        "token": token,
        "user": {"id": user.id, "email": user.email}
    }), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    user = User.query.filter_by(email=email).first()
    if not user or not bcrypt.check_password_hash(user.password_hash, password):
        return jsonify({"error": "Invalid email or password"}), 401

    token = create_access_token(identity=str(user.id))
    return jsonify({
        "message": "Login successful",
        "token": token,
        "user": {"id": user.id, "email": user.email}
    }), 200


@auth_bp.route("/logout", methods=["POST"])
@jwt_required()
def logout():
    jti = get_jwt()["jti"]
    BLOCKLISTED_TOKENS.add(jti)
    return jsonify({"message": "Successfully logged out"}), 200


@auth_bp.route("/me", methods=["GET"])
@jwt_required()
def me():
    from flask_jwt_extended import get_jwt_identity
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify({"id": user.id, "email": user.email}), 200
