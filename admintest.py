from app import app, db
from models import User

with app.app_context():
    user = User.query.filter_by(email="makeskilled@gmail.com").first()
    if user:
        user.role = "admin"
        db.session.commit()
        print("User promoted to admin!")