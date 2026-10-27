from pathlib import Path
from functools import wraps
import logging
import secrets
import uuid
import json
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request, session, send_from_directory
from flask_cors import CORS
from sqlalchemy import func, text
from .config import Config
from .extensions import db
from .models import User, Analysis, Feedback
from .services.image_utils import validate_image, ImageValidationError, image_quality, detect_face
from .services.model_service import PredictionService, ModelUnavailable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deepsafe")

def create_app(test_config=None):
    app = Flask(__name__, static_folder="../frontend", static_url_path="")
    app.config.from_object(Config)
    if test_config:
        app.config.update(test_config)
    app.config["MAX_CONTENT_LENGTH"] = int(app.config.get("MAX_CONTENT_LENGTH", 10 * 1024 * 1024))
    db.init_app(app)
    allowed = [x.strip() for x in __import__("os").getenv("CORS_ORIGINS", "http://127.0.0.1:5000,http://localhost:5000").split(",") if x.strip()]
    CORS(app, supports_credentials=True, origins=allowed)
    app.config["UPLOAD_DIR"].mkdir(parents=True, exist_ok=True)
    (app.config["UPLOAD_DIR"] / "gradcam").mkdir(parents=True, exist_ok=True)
    service = PredictionService(app.config["MODEL_PATH"], app.config["METADATA_PATH"], app.config["UPLOAD_DIR"] / "gradcam")
    app.extensions["prediction_service"] = service

    def error(message, status=400):
        return jsonify(success=False, error=message), status

    def login_required(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            if not session.get("user_id"):
                return error("Please log in to continue.", 401)
            return fn(*args, **kwargs)
        return wrapped

    def admin_required(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            user = db.session.get(User, session.get("user_id")) if session.get("user_id") else None
            if not user or user.role != "admin":
                return error("Administrator access is required.", 403)
            return fn(*args, **kwargs)
        return wrapped

    @app.errorhandler(413)
    def too_large(_):
        return error("Image is too large. Reduce its size and try again.", 413)

    @app.get("/")
    def index():
        return app.send_static_file("index.html")

    @app.get("/api/health")
    def health():
        database_ok = False
        try:
            db.session.execute(text("SELECT 1"))
            database_ok = True
        except Exception:
            db.session.rollback()
        return jsonify(success=True, service="DeepSafe", database="connected" if database_ok else "unavailable",
                       model="available" if service.available() else "missing",
                       model_message=None if service.available() else "Trained model is not available. Please train the model first.")

    @app.post("/api/auth/signup")
    def signup():
        data = request.get_json(silent=True) or {}
        name, email, password = (str(data.get(k, "")).strip() for k in ("name", "email", "password"))
        if not name or len(name) > 120 or "@" not in email or len(email) > 254 or len(password) < 8:
            return error("Provide a name, valid email, and password with at least 8 characters.")
        if User.query.filter(func.lower(User.email) == email.lower()).first():
            return error("An account with this email already exists.", 409)
        user = User(name=name, email=email.lower())
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        session.clear()
        session["user_id"] = user.id
        session["csrf_token"] = secrets.token_urlsafe(24)
        return jsonify(success=True, user=user.public()), 201

    @app.post("/api/auth/login")
    def login():
        data = request.get_json(silent=True) or {}
        email = str(data.get("email", "")).strip().lower()
        password = str(data.get("password", ""))
        user = User.query.filter_by(email=email).first()
        if not user or not user.check_password(password):
            return error("Email or password is incorrect.", 401)
        session.clear()
        session["user_id"] = user.id
        session["csrf_token"] = secrets.token_urlsafe(24)
        return jsonify(success=True, user=user.public())

    @app.post("/api/auth/logout")
    def logout():
        session.clear()
        return jsonify(success=True, message="Logged out.")

    @app.get("/api/auth/me")
    def me():
        user = db.session.get(User, session.get("user_id")) if session.get("user_id") else None
        return jsonify(success=True, user=user.public() if user else None)

    @app.post("/api/analyze/image")
    @login_required
    def analyze_image():
        try:
            raw = validate_image(request.files.get("image"), app.config["MAX_CONTENT_LENGTH"])
            image_id = uuid.uuid4().hex
            upload_name = f"{image_id}.jpg"
            upload_path = app.config["UPLOAD_DIR"] / upload_name
            pil_image = Image.open(__import__("io").BytesIO(raw)).convert("RGB")
            pil_image.save(upload_path, format="JPEG", quality=95)
            arr = np.asarray(pil_image)
            quality, width, height = image_quality(arr)
            face_detected, face_count = detect_face(arr)
            try:
                result = service.predict(upload_path, image_id)
            except ModelUnavailable as exc:
                upload_path.unlink(missing_ok=True)
                return error(str(exc), 503)
            except Exception:
                logger.exception("Inference failed")
                upload_path.unlink(missing_ok=True)
                return error("Model inference or explanation generation failed. Check the backend log and trained model.", 500)
            grad_rel = str(Path("uploads") / "gradcam" / f"{image_id}_gradcam.jpg")
            overlay_rel = str(Path("uploads") / "gradcam" / f"{image_id}_overlay.jpg")
            row = Analysis(user_id=session["user_id"], image_path=str(Path("uploads") / upload_name),
                           prediction=result["prediction"], confidence=result["confidence"],
                           real_probability=result["real_probability"], deepfake_probability=result["deepfake_probability"],
                           image_quality=quality, face_detected=face_detected,
                           model_version=result["model_version"], gradcam_path=grad_rel)
            db.session.add(row)
            db.session.commit()
            payload = row.public()
            payload.update(success=True, image_width=width, image_height=height, face_count=face_count,
                           overlay_url="/" + overlay_rel.replace("\\\\", "/"),
                           disclaimer="AI-based detection is probabilistic and should not be treated as absolute proof.")
            return jsonify(payload), 201
        except ImageValidationError as exc:
            return error(str(exc))
        except Exception:
            db.session.rollback()
            logger.exception("Image analysis failed")
            return error("Could not analyze this image. Check the file and database connection.", 500)

    @app.get("/api/history")
    @login_required
    def history():
        rows = Analysis.query.filter_by(user_id=session["user_id"]).order_by(Analysis.created_at.desc()).limit(200).all()
        return jsonify(success=True, results=[r.public() for r in rows])

    @app.get("/api/history/<int:analysis_id>")
    @login_required
    def history_detail(analysis_id):
        row = Analysis.query.filter_by(id=analysis_id, user_id=session["user_id"]).first()
        if not row:
            return error("Analysis was not found.", 404)
        return jsonify(success=True, result=row.public())

    @app.delete("/api/history/<int:analysis_id>")
    @login_required
    def history_delete(analysis_id):
        row = Analysis.query.filter_by(id=analysis_id, user_id=session["user_id"]).first()
        if not row:
            return error("Analysis was not found.", 404)
        db.session.delete(row)
        db.session.commit()
        return jsonify(success=True, message="Analysis deleted.")

    @app.post("/api/feedback")
    @login_required
    def feedback():
        data = request.get_json(silent=True) or {}
        category, message = str(data.get("category", "")).strip(), str(data.get("message", "")).strip()
        if category not in {"Report a Problem", "Wrong Prediction", "Technical Issue", "Feedback", "Other"}:
            return error("Choose a valid feedback category.")
        if not message or len(message) > 5000:
            return error("Enter a message of up to 5000 characters.")
        row = Feedback(user_id=session["user_id"], category=category, message=message)
        db.session.add(row)
        db.session.commit()
        return jsonify(success=True, message="Thank you. Your feedback was saved."), 201

    @app.get("/api/admin/summary")
    @admin_required
    def admin_summary():
        try:
            total_users = User.query.count()
            total_analyses = Analysis.query.count()
            real_count = Analysis.query.filter_by(prediction="REAL").count()
            fake_count = Analysis.query.filter_by(prediction="DEEPFAKE").count()
            feedback_count = Feedback.query.count()
            recent = db.session.query(func.date(Analysis.created_at), func.count(Analysis.id)).group_by(func.date(Analysis.created_at)).order_by(func.date(Analysis.created_at)).limit(30).all()
            metadata = service.metadata
            return jsonify(success=True, totals={"users": total_users, "analyses": total_analyses,
                "real": real_count, "deepfake": fake_count, "feedback": feedback_count},
                analyses_over_time=[{"date": str(day), "count": count} for day, count in recent],
                model=metadata)
        except Exception:
            logger.exception("Admin summary failed")
            return error("Could not load admin data. Verify database availability.", 503)

    @app.get("/uploads/<path:filename>")
    def uploaded_file(filename):
        # Only generated, non-user-controlled filenames are written under this directory.
        return send_from_directory(app.config["UPLOAD_DIR"], filename)

    return app

app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
