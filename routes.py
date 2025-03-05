from flask import render_template, redirect, url_for, flash, request,jsonify,Response
from app import app, db,bcrypt
from models import User,YogaPose  # Import the User model
from flask_login import login_user, logout_user, login_required, current_user
from models import YogaPose
import os
import json
import cv2
import numpy as np
import mediapipe as mp
from werkzeug.utils import secure_filename

from models import UserAsanaUsage  # Import relevant models
from datetime import datetime, timedelta
import csv


UPLOAD_FOLDER = 'static/poses'  # Folder to store uploaded pose images
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure folder exists

mp_pose = mp.solutions.pose

def extract_pose_angles(image_path):
    """Extract key joint angles from a given pose image using MediaPipe."""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        keypoints = {i: (lm.x, lm.y) for i, lm in enumerate(results.pose_landmarks.landmark)}

        def calculate_angle(a, b, c):
            """Compute the angle between three points."""
            a, b, c = np.array(a), np.array(b), np.array(c)
            ba = a - b
            bc = c - b
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            return np.degrees(np.arccos(cosine_angle))

        joint_angles = {
            "left_elbow": calculate_angle(keypoints[11], keypoints[13], keypoints[15]),
            "right_elbow": calculate_angle(keypoints[12], keypoints[14], keypoints[16]),
            "left_knee": calculate_angle(keypoints[23], keypoints[25], keypoints[27]),
            "right_knee": calculate_angle(keypoints[24], keypoints[26], keypoints[28])
        }

        json_path = image_path.replace('.jpg', '.json').replace('.png', '.json')
        with open(json_path, "w") as f:
            json.dump(joint_angles, f)

        return joint_angles
    return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard')) if user.role == "user" else redirect(url_for('admin'))
        flash("Invalid email or password", "danger")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
        user = User(username=username, email=email, password=password)
        db.session.add(user)
        db.session.commit()
        flash("Account created! You can now log in.", "success")
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    poses = YogaPose.query.all()  # Fetch all yoga poses
    print("Fetched Poses:", poses)  # Debugging: Check if poses are fetched
    return render_template('dashboard.html', user=current_user, poses=poses)

@app.route('/admin', methods=['GET', 'POST'])
@login_required
def admin():
    if current_user.role != 'admin':
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        name = request.form['name']
        description = request.form['description']

        if 'reference_pose' not in request.files:
            flash("No image uploaded!", "danger")
            return redirect(url_for('admin'))

        file = request.files['reference_pose']
        if file.filename == '':
            flash("No selected file!", "danger")
            return redirect(url_for('admin'))

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)  # Save the uploaded file
        print(f"✅ Pose image saved at: {file_path}")

        # Extract pose angles
        joint_angles = extract_pose_angles(file_path)

        if joint_angles:
            print(f"✅ Extracted Pose Angles: {joint_angles}")
            print(file_path)
            new_pose = YogaPose(name=name, description=description, reference_pose=file_path)

            db.session.add(new_pose)
            db.session.commit()
            print("✅ Pose added to the database!")
            flash("Yoga pose added successfully with AI pose detection!", "success")
        else:
            print("❌ Pose detection failed!")
            flash("Pose detection failed. Please upload a clear image!", "danger")

    poses = YogaPose.query.all()
    return render_template('admin.html', user=current_user, poses=poses)

import cv2
import mediapipe as mp
import numpy as np
import json
import os

mp_pose = mp.solutions.pose

@app.route('/edit_asana/<int:asana_id>', methods=['GET', 'POST'])
@login_required
def edit_asana(asana_id):
    if current_user.role != 'admin':
        return redirect(url_for('dashboard'))

    asana = YogaPose.query.get_or_404(asana_id)

    if request.method == 'POST':
        asana.name = request.form['name']
        asana.description = request.form['description']

        # Handle reference pose image update
        if 'reference_pose' in request.files and request.files['reference_pose'].filename != '':
            file = request.files['reference_pose']
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # ✅ Update the reference pose path in the database
            asana.reference_pose = file_path

            # ✅ Extract new pose angles and update JSON
            pose_angles = extract_pose_angles(file_path)
            if pose_angles:
                json_filename = f"{asana.name.replace(' ', '_')}.json"
                json_path = os.path.join(app.config['UPLOAD_FOLDER'], json_filename)

                with open(json_path, "w") as f:
                    json.dump(pose_angles, f)

                print(f"✅ Updated Pose Tracking Data: {json_path}")
            else:
                flash("Pose detection failed! Try uploading a clearer image.", "danger")

        db.session.commit()
        flash("Asana updated successfully!", "success")
        return redirect(url_for('admin'))
    print(asana.reference_pose)
    return render_template('edit_asana.html', asana=asana)

@app.route('/delete_asana/<int:asana_id>', methods=['POST'])
@login_required
def delete_asana(asana_id):
    if current_user.role != 'admin':
        return redirect(url_for('dashboard'))

    asana = YogaPose.query.get_or_404(asana_id)

    # Delete reference pose image from storage
    if os.path.exists(asana.reference_pose):
        os.remove(asana.reference_pose)

    db.session.delete(asana)
    db.session.commit()
    flash("Asana deleted successfully!", "danger")

    return redirect(url_for('admin'))


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/pose_tracking/<int:pose_id>')
@login_required
def pose_tracking(pose_id):
    pose = YogaPose.query.get_or_404(pose_id)
    print()

    # Get the latest ongoing session for the user
    last_usage = UserAsanaUsage.query.filter_by(user_id=current_user.id, asana_id=pose_id, end_time=None).first()

    return render_template('pose_tracking.html', pose=pose, last_usage_id=last_usage.id if last_usage else None,x=pose.name.strip())


@app.route('/admin_dashboard')
@login_required
def admin_dashboard():
    if current_user.role != 'admin':
        return redirect(url_for('dashboard'))  # Prevent non-admin access

    total_users = User.query.count()
    total_asanas = YogaPose.query.count()
    
    # Count how many times each asana has been used
    asana_usage = db.session.query(
        YogaPose.name, db.func.count(UserAsanaUsage.asana_id)
    ).join(UserAsanaUsage, YogaPose.id == UserAsanaUsage.asana_id).group_by(YogaPose.name).all()
    
    # Calculate total time spent by users on asanas
    total_time_spent = db.session.query(
        db.func.sum(UserAsanaUsage.time_spent)
    ).scalar() or 0  # Default to 0 if no records

    return render_template('admin_dashboard.html', 
                           total_users=total_users, 
                           total_asanas=total_asanas, 
                           asana_usage=asana_usage, 
                           total_time_spent=total_time_spent)

@app.route('/start_asana/<int:asana_id>')
@login_required
def start_asana(asana_id):
    # Create a new session record for this user and asana
    usage = UserAsanaUsage(user_id=current_user.id, asana_id=asana_id)
    db.session.add(usage)
    db.session.commit()

    # Redirect to pose tracking page
    return redirect(url_for('pose_tracking', pose_id=asana_id))


@app.route('/stop_asana/<int:usage_id>')
@login_required
def stop_asana(usage_id):
    usage = UserAsanaUsage.query.get(usage_id)
    if usage and usage.user_id == current_user.id:
        usage.stop_tracking()
        db.session.commit()
    
    return redirect(url_for('dashboard'))

@app.route('/history')
@login_required
def history():
    history = UserAsanaUsage.query.filter_by(user_id=current_user.id).order_by(UserAsanaUsage.start_time.desc()).all()
    return render_template('user_history.html', history=history)

@app.route('/leaderboard')
@login_required
def leaderboard():
    # Query to get total time spent by each user
    user_stats = db.session.query(
        User.username, 
        db.func.count(UserAsanaUsage.id).label("asana_count"),
        db.func.sum(UserAsanaUsage.time_spent).label("total_time_spent")
    ).join(UserAsanaUsage).group_by(User.id).order_by(db.desc("total_time_spent")).limit(10).all()

    return render_template('leaderboard.html', user_stats=user_stats)


@app.route('/user_performance_data')
@login_required
def user_performance_data():
    # Query top 5 users by total time spent
    user_stats = db.session.query(
        User.id,
        User.username,
        db.func.sum(UserAsanaUsage.time_spent).label("total_time_spent")
    ).join(UserAsanaUsage).group_by(User.id).order_by(db.desc("total_time_spent")).limit(5).all()

    # Extract usernames and time spent correctly
    top_users = [user.username for user in user_stats]
    time_spent = [int(user.total_time_spent // 60) if user.total_time_spent else 0 for user in user_stats]

    # Assign badges based on ranking
    badges = ["🥇 Gold", "🥈 Silver", "🥉 Bronze"] + [""] * (len(user_stats) - 3)

    # Query daily practice for logged-in user
    daily_progress = db.session.query(
        db.func.date(UserAsanaUsage.start_time).label("date"),
        db.func.sum(UserAsanaUsage.time_spent).label("daily_time_spent")
    ).filter(UserAsanaUsage.user_id == current_user.id).group_by("date").all()

    dates = [str(date) for date, _ in daily_progress]
    daily_time = [int(time_spent // 60) if time_spent else 0 for _, time_spent in daily_progress]

    # Check for extra badges
    active_learner = "🔥 Active Learner" if len(daily_progress) >= 5 else ""
    consistency_star = "🌟 Consistency Star" if len(dates) >= 3 and (
        (datetime.strptime(dates[-1], "%Y-%m-%d") - datetime.strptime(dates[-2], "%Y-%m-%d")).days == 1 and
        (datetime.strptime(dates[-2], "%Y-%m-%d") - datetime.strptime(dates[-3], "%Y-%m-%d")).days == 1
    ) else ""

    return jsonify({
        "top_users": top_users,
        "time_spent": time_spent,
        "badges": badges,
        "dates": dates,
        "daily_time": daily_time,
        "extra_badges": {"active_learner": active_learner, "consistency_star": consistency_star}
    })


@app.route('/admin_leaderboard')
@login_required
def admin_leaderboard():
    if current_user.role != 'admin':
        return redirect(url_for('dashboard'))  # Prevent non-admins from accessing

    # Query top users
    user_stats = db.session.query(
        User.id,
        User.username,
        db.func.count(UserAsanaUsage.id).label("asana_count"),
        db.func.sum(UserAsanaUsage.time_spent).label("total_time_spent")
    ).join(UserAsanaUsage).group_by(User.id).order_by(db.desc("total_time_spent")).limit(10).all()

    return render_template('admin_leaderboard.html', user_stats=user_stats)


@app.route('/export_leaderboard_csv')
@login_required
def export_leaderboard_csv():
    if current_user.role != 'admin':
        return redirect(url_for('dashboard'))  # Restrict access to admin only

    # Query top 10 users
    user_stats = db.session.query(
        User.username,
        db.func.count(UserAsanaUsage.id).label("asana_count"),
        db.func.sum(UserAsanaUsage.time_spent).label("total_time_spent")
    ).join(UserAsanaUsage).group_by(User.id).order_by(db.desc("total_time_spent")).limit(10).all()

    # Generate CSV content
    csv_data = "Username,Total Asanas,Total Time Spent (min)\n"
    for user, asana_count, total_time_spent in user_stats:
        csv_data += f"{user},{asana_count},{int(total_time_spent // 60) if total_time_spent else 0}\n"

    # Return as downloadable file
    response = Response(csv_data, mimetype="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=leaderboard.csv"
    return response

@app.route('/admin_analytics_data')
@login_required
def admin_analytics_data():
    if current_user.role != 'admin':
        return redirect(url_for('dashboard'))  # Restrict access to admin only

    # ✅ Correct usage of `datetime.utcnow()`
    last_week = db.session.query(
        db.func.date(UserAsanaUsage.start_time).label("date"),
        db.func.sum(UserAsanaUsage.time_spent).label("time_spent")
    ).filter(UserAsanaUsage.start_time >= datetime.utcnow() - timedelta(days=7)) \
     .group_by("date").all()

    weekly_dates = [str(date) for date, _ in last_week]
    weekly_times = [int(time_spent // 60) if time_spent else 0 for _, time_spent in last_week]

    last_month = db.session.query(
        db.func.date(UserAsanaUsage.start_time).label("date"),
        db.func.sum(UserAsanaUsage.time_spent).label("time_spent")
    ).filter(UserAsanaUsage.start_time >= datetime.utcnow() - timedelta(days=30)) \
     .group_by("date").all()

    monthly_dates = [str(date) for date, _ in last_month]
    monthly_times = [int(time_spent // 60) if time_spent else 0 for _, time_spent in last_month]

    return jsonify({
        "weekly_dates": weekly_dates,
        "weekly_times": weekly_times,
        "monthly_dates": monthly_dates,
        "monthly_times": monthly_times
    })
