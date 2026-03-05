"""
Flask Application Factory
"""
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager
from flask_bcrypt import Bcrypt
from flask_cors import CORS

db = SQLAlchemy()
jwt = JWTManager()
bcrypt = Bcrypt()

# JWT blocklist (in-memory; replace with Redis/DB for production)
BLOCKLISTED_TOKENS = set()


def create_app():
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), '..', 'frontend', 'templates'),
        static_folder=os.path.join(os.path.dirname(__file__), '..', 'frontend', 'static'),
    )

    from config import Config
    app.config.from_object(Config)

    # Ensure required directories exist
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'models'), exist_ok=True)

    # Extensions
    db.init_app(app)
    jwt.init_app(app)
    bcrypt.init_app(app)
    CORS(app, supports_credentials=True)

    # JWT blocklist check
    @jwt.token_in_blocklist_loader
    def check_if_token_revoked(jwt_header, jwt_payload):
        jti = jwt_payload["jti"]
        return jti in BLOCKLISTED_TOKENS

    # Register Blueprints
    from app.auth import auth_bp
    from app.upload import upload_bp
    from app.pipeline import pipeline_bp
    from app.eda import eda_bp
    from app.views import views_bp

    app.register_blueprint(auth_bp, url_prefix="/api")
    app.register_blueprint(upload_bp, url_prefix="/api")
    app.register_blueprint(pipeline_bp, url_prefix="/api")
    app.register_blueprint(eda_bp, url_prefix="/api")
    app.register_blueprint(views_bp)

    with app.app_context():
        db.create_all()

    return app
