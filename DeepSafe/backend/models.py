from datetime import datetime, timezone
from werkzeug.security import generate_password_hash, check_password_hash
from .extensions import db

def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)

class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(254), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="user")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def public(self):
        return {"id": self.id, "name": self.name, "email": self.email, "role": self.role}

class Analysis(db.Model):
    __tablename__ = "analyses"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    image_path = db.Column(db.String(500), nullable=False)
    prediction = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    real_probability = db.Column(db.Float, nullable=False)
    deepfake_probability = db.Column(db.Float, nullable=False)
    image_quality = db.Column(db.String(20), nullable=False)
    face_detected = db.Column(db.Boolean, nullable=False)
    model_version = db.Column(db.String(40), nullable=False)
    gradcam_path = db.Column(db.String(500), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)

    def public(self):
        return {
            "id": self.id, "prediction": self.prediction, "confidence": round(self.confidence, 2),
            "probabilities": {"REAL": round(self.real_probability, 2), "DEEPFAKE": round(self.deepfake_probability, 2)},
            "image_quality": self.image_quality, "face_detected": self.face_detected,
            "model_version": self.model_version, "created_at": self.created_at.isoformat() + "Z",
            "image_url": "/" + self.image_path.replace("\\\\", "/"),
            "gradcam_url": "/" + self.gradcam_path.replace("\\\\", "/")
        }

class Feedback(db.Model):
    __tablename__ = "feedback"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    category = db.Column(db.String(50), nullable=False)
    message = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
