from app import app, db, login_manager
from flask_login import UserMixin
from datetime import datetime

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    role = db.Column(db.String(10), nullable=False, default="user")  # admin or user

class YogaPose(db.Model):
    __tablename__ = 'yogapose'  # ✅ Ensure this matches exactly
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)
    reference_pose = db.Column(db.String(255), nullable=False)


class UserAsanaUsage(db.Model):
    __tablename__ = 'user_asana_usage'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))  # ✅ Ensure this matches the User table
    asana_id = db.Column(db.Integer, db.ForeignKey('yogapose.id'))  # ✅ Matches YogaPose table
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime, nullable=True)
    time_spent = db.Column(db.Integer, nullable=True)  # Time in seconds

    # ✅ Establish relationship with YogaPose
    asana = db.relationship('YogaPose', backref='asana_usages')

    def stop_tracking(self):
        self.end_time = datetime.utcnow()
        self.time_spent = (self.end_time - self.start_time).total_seconds()


# Ensure the database is created inside the application context
with app.app_context():
    db.create_all()
