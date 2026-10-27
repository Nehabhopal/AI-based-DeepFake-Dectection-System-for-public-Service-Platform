import pytest
from backend.app import create_app
from backend.extensions import db

@pytest.fixture()
def app(tmp_path):
    app = create_app({
        "TESTING": True,
        "SECRET_KEY": "test-secret",
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'test.db'}",
        "SQLALCHEMY_TRACK_MODIFICATIONS": False,
        "MAX_CONTENT_LENGTH": 1024 * 1024,
        "UPLOAD_DIR": tmp_path / "uploads",
        "MODEL_PATH": tmp_path / "missing.pth",
        "METADATA_PATH": tmp_path / "metadata.json",
    })
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()

@pytest.fixture()
def client(app):
    return app.test_client()
