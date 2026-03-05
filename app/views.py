"""
Views — Serves the HTML frontend pages
"""
from flask import Blueprint, render_template

views_bp = Blueprint("views", __name__)


@views_bp.route("/")
def index():
    return render_template("login.html")


@views_bp.route("/register")
def register_page():
    return render_template("register.html")


@views_bp.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@views_bp.route("/results")
def results():
    return render_template("results.html")
