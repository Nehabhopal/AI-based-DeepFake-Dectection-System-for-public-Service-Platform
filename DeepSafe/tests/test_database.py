from backend.models import User, Analysis, Feedback
from backend.extensions import db

def test_user_password_hash_and_feedback_storage(app):
    with app.app_context():
        u=User(name="Example",email="example@example.com")
        u.set_password("strong-password")
        db.session.add(u); db.session.commit()
        assert u.password_hash!="strong-password"
        assert u.check_password("strong-password")
        f=Feedback(user_id=u.id,category="Feedback",message="Test message")
        db.session.add(f); db.session.commit()
        assert Feedback.query.count()==1
