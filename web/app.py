from datetime import date, datetime, timezone
from functools import wraps
import logging
import os
from uuid import uuid4

from dotenv import load_dotenv
from flask import (
    Flask,
    abort,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Boolean, func
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError
from sqlalchemy.exc import SQLAlchemyError

load_dotenv()

from config import Config
from services.ai_service import analyze_dental_image
from services.recommendation_service import general_guidance

db = SQLAlchemy()
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
REVIEW_STATUSES = {"Reviewed", "Follow-up Required", "Completed"}


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    phone = db.Column(db.String(40))
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class Screening(db.Model):
    __tablename__ = "screenings"

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    image_filename = db.Column(db.String(255), nullable=False)
    ai_result = db.Column(db.String(120), nullable=False)
    confidence = db.Column(db.Float)
    explanation = db.Column(db.Text, nullable=False)
    recommendation = db.Column(db.Text, nullable=False)
    model_mode = db.Column(db.String(30), nullable=False, default="demo")
    review_status = db.Column(db.String(30), nullable=False, default="Pending")
    review_requested = db.Column(Boolean, nullable=False, default=False)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    patient = db.relationship("User", foreign_keys=[patient_id])
    dentist_review = db.relationship(
        "DentistReview", back_populates="screening", uselist=False, cascade="all, delete-orphan"
    )


class DentistReview(db.Model):
    __tablename__ = "dentist_reviews"

    id = db.Column(db.Integer, primary_key=True)
    screening_id = db.Column(db.Integer, db.ForeignKey("screenings.id"), nullable=False, unique=True)
    dentist_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    professional_finding = db.Column(db.Text, nullable=False)
    treatment_plan = db.Column(db.Text, nullable=False)
    notes = db.Column(db.Text)
    follow_up_date = db.Column(db.Date)
    review_status = db.Column(db.String(30), nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    dentist = db.relationship("User", foreign_keys=[dentist_id])
    screening = db.relationship("Screening", back_populates="dentist_review")


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(Config)
    if test_config:
        app.config.update(test_config)
    os.makedirs(app.instance_path, exist_ok=True)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    db.init_app(app)

    logging.basicConfig(level=logging.INFO)
    with app.app_context():
        try:
            db.create_all()
        except SQLAlchemyError:
            app.logger.exception("Unable to initialize the database")
            raise

    def login_required(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if "user_id" not in session:
                flash("Please log in to continue.", "warning")
                return redirect(url_for("login"))
            return view(*args, **kwargs)
        return wrapped

    def role_required(role):
        def decorator(view):
            @wraps(view)
            @login_required
            def wrapped(*args, **kwargs):
                if session.get("role") != role:
                    abort(403)
                return view(*args, **kwargs)
            return wrapped
        return decorator

    def get_screening_for_patient(screening_id):
        screening = db.get_or_404(Screening, screening_id)
        if screening.patient_id != session.get("user_id"):
            abort(404)
        return screening

    @app.context_processor
    def inject_current_user():
        user = (
            {"name": session.get("user_name"), "role": session.get("role")}
            if session.get("user_id")
            else None
        )
        return {"current_user": user}

    def commit_changes(error_message):
        try:
            db.session.commit()
            return True
        except SQLAlchemyError:
            db.session.rollback()
            app.logger.exception("Database write failed")
            flash(error_message, "danger")
            return False

    @app.errorhandler(413)
    def file_too_large(_error):
        flash("That image is too large. Choose an image under 8 MB.", "danger")
        return redirect(url_for("upload_image"))

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if request.method == "POST":
            name = request.form.get("name", "").strip()
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            role = request.form.get("role", "").strip().lower()
            phone = request.form.get("phone", "").strip() or None
            if not name or len(name) > 120 or not email or len(email) > 255:
                flash("Enter a valid name and email address.", "danger")
            elif len(password) < 8:
                flash("Choose a password with at least 8 characters.", "danger")
            elif role not in {"patient", "dentist"}:
                flash("Choose a valid account type.", "danger")
            elif db.session.scalar(db.select(User).filter_by(email=email)):
                flash("An account with that email already exists.", "danger")
            else:
                user = User(
                    name=name,
                    email=email,
                    password_hash=generate_password_hash(password),
                    role=role,
                    phone=phone,
                )
                db.session.add(user)
                if not commit_changes("Your account could not be created. Please try again."):
                    return render_template("register.html")
                session.clear()
                session["user_id"] = user.id
                session["role"] = user.role
                session["user_name"] = user.name
                flash("Your account is ready.", "success")
                return redirect(url_for("patient_dashboard" if role == "patient" else "dentist_dashboard"))
        return render_template("register.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            user = db.session.scalar(db.select(User).filter_by(email=email))
            if user is None or not check_password_hash(user.password_hash, password):
                flash("Email or password was not recognized.", "danger")
            else:
                session.clear()
                session["user_id"] = user.id
                session["role"] = user.role
                session["user_name"] = user.name
                return redirect(url_for("patient_dashboard" if user.role == "patient" else "dentist_dashboard"))
        return render_template("login.html")

    @app.route("/logout", methods=["POST"])
    def logout():
        session.clear()
        flash("You have been logged out.", "success")
        return redirect(url_for("index"))

    @app.route("/patient")
    @role_required("patient")
    def patient_dashboard():
        patient_id = session["user_id"]
        screenings = db.session.scalars(
            db.select(Screening).filter_by(patient_id=patient_id).order_by(Screening.created_at.desc())
        ).all()
        pending_count = sum(
            1 for item in screenings if item.review_requested and item.review_status == "Pending"
        )
        follow_up = db.session.scalar(
            db.select(DentistReview.follow_up_date)
            .join(Screening)
            .where(Screening.patient_id == patient_id, DentistReview.follow_up_date >= date.today())
            .order_by(DentistReview.follow_up_date)
        )
        return render_template(
            "patient/dashboard.html",
            screenings=screenings[:5],
            total_screenings=len(screenings),
            latest=screenings[0] if screenings else None,
            pending_count=pending_count,
            next_follow_up=follow_up,
            chart_labels=[item.created_at.strftime("%b %d") for item in reversed(screenings[:8])],
            chart_values=[1 for _item in reversed(screenings[:8])],
        )

    @app.route("/patient/upload", methods=["GET", "POST"])
    @role_required("patient")
    def upload_image():
        if request.method == "POST":
            uploaded = request.files.get("image")
            if uploaded is None or not uploaded.filename:
                flash("Choose an image before submitting.", "danger")
                return render_template("patient/upload.html")
            filename = secure_filename(uploaded.filename)
            extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            if extension not in ALLOWED_EXTENSIONS:
                flash("Upload a JPG, JPEG, or PNG image.", "danger")
                return render_template("patient/upload.html")
            stored_name = f"{uuid4().hex}.{extension}"
            destination = os.path.join(app.config["UPLOAD_FOLDER"], stored_name)
            try:
                uploaded.stream.seek(0)
                with Image.open(uploaded.stream) as image:
                    image.verify()
                uploaded.stream.seek(0)
                uploaded.save(destination)
            except (UnidentifiedImageError, OSError, ValueError):
                flash("The selected file is not a readable image. Try another JPG or PNG.", "danger")
                return render_template("patient/upload.html")
            try:
                result = analyze_dental_image(destination)
            except Exception:
                app.logger.exception("Dental image analysis failed")
                try:
                    os.remove(destination)
                except OSError:
                    app.logger.exception("Could not remove an image after analysis failed")
                flash("Image screening could not be completed. Please try again.", "danger")
                return render_template("patient/upload.html")
            screening = Screening(
                patient_id=session["user_id"],
                image_filename=stored_name,
                ai_result=result["condition"],
                confidence=result.get("confidence"),
                explanation=result["explanation"],
                recommendation=result["recommendation"],
                model_mode=result["mode"],
            )
            db.session.add(screening)
            try:
                db.session.commit()
            except SQLAlchemyError:
                db.session.rollback()
                app.logger.exception("Could not save screening to the database")
                try:
                    os.remove(destination)
                except OSError:
                    app.logger.exception("Could not remove an image after its screening failed to save")
                flash("The screening could not be saved. Please try again.", "danger")
                return render_template("patient/upload.html")
            return redirect(url_for("screening_result", screening_id=screening.id))
        return render_template("patient/upload.html")

    @app.route("/patient/history")
    @role_required("patient")
    def patient_history():
        screenings = db.session.scalars(
            db.select(Screening)
            .filter_by(patient_id=session["user_id"])
            .order_by(Screening.created_at.desc())
        ).all()
        return render_template("patient/history.html", screenings=screenings)

    @app.route("/patient/reviews")
    @role_required("patient")
    def patient_reviews():
        screenings = db.session.scalars(
            db.select(Screening)
            .where(
                Screening.patient_id == session["user_id"],
                Screening.review_requested.is_(True),
            )
            .order_by(Screening.created_at.desc())
        ).all()
        return render_template("patient/reviews.html", screenings=screenings)

    @app.route("/patient/screenings/<int:screening_id>")
    @role_required("patient")
    def screening_result(screening_id):
        screening = get_screening_for_patient(screening_id)
        return render_template(
            "patient/result.html",
            screening=screening,
            guidance=general_guidance(),
        )

    @app.route("/patient/screenings/<int:screening_id>/request-review", methods=["POST"])
    @role_required("patient")
    def request_review(screening_id):
        screening = get_screening_for_patient(screening_id)
        if screening.dentist_review:
            flash("A dentist has already reviewed this screening.", "info")
            return redirect(url_for("screening_result", screening_id=screening.id))
        if screening.review_requested:
            flash("This screening is already in the dentist review queue.", "info")
            return redirect(url_for("screening_result", screening_id=screening.id))
        screening.review_status = "Pending"
        screening.review_requested = True
        if not commit_changes("Your review request could not be saved. Please try again."):
            return redirect(url_for("screening_result", screening_id=screening.id))
        flash("Your image has been added to the dentist review queue.", "success")
        return redirect(url_for("screening_result", screening_id=screening.id))

    @app.route("/patient/screenings/<int:screening_id>/image")
    @login_required
    def screening_image(screening_id):
        screening = db.get_or_404(Screening, screening_id)
        if session.get("role") not in {"patient", "dentist"}:
            abort(403)
        if session.get("role") == "patient" and screening.patient_id != session.get("user_id"):
            abort(404)
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], screening.image_filename)
        if not os.path.isfile(image_path):
            abort(404)
        return send_file(
            image_path,
            mimetype="image/jpeg" if screening.image_filename.endswith((".jpg", ".jpeg")) else "image/png",
        )

    @app.route("/patient/profile", methods=["GET", "POST"])
    @role_required("patient")
    def patient_profile():
        user = db.get_or_404(User, session["user_id"])
        if request.method == "POST":
            name = request.form.get("name", "").strip()
            phone = request.form.get("phone", "").strip()
            if not name or len(name) > 120 or len(phone) > 40:
                flash("Enter a valid name and phone number.", "danger")
            else:
                user.name = name
                user.phone = phone or None
                if not commit_changes("Your profile could not be updated. Please try again."):
                    return render_template("patient/profile.html", user=user)
                session["user_name"] = name
                flash("Profile updated.", "success")
                return redirect(url_for("patient_profile"))
        return render_template("patient/profile.html", user=user)

    @app.route("/dentist")
    @role_required("dentist")
    def dentist_dashboard():
        screenings = db.session.scalars(
            db.select(Screening)
            .where(Screening.review_requested.is_(True))
            .order_by(Screening.created_at.desc())
        ).all()
        return render_template(
            "dentist/dashboard.html",
            screenings=screenings,
            total_patients=db.session.scalar(db.select(func.count(func.distinct(Screening.patient_id)))) or 0,
            pending_count=sum(1 for item in screenings if item.review_status == "Pending"),
            completed_count=sum(1 for item in screenings if item.review_status == "Completed"),
            follow_up_count=db.session.scalar(
                db.select(func.count(DentistReview.id)).where(
                    DentistReview.follow_up_date >= date.today()
                )
            ) or 0,
        )

    @app.route("/dentist/screenings/<int:screening_id>", methods=["GET", "POST"])
    @role_required("dentist")
    def dentist_review(screening_id):
        screening = db.get_or_404(Screening, screening_id)
        if not screening.review_requested and not screening.dentist_review:
            abort(404)
        review = screening.dentist_review
        if request.method == "POST":
            finding = request.form.get("professional_finding", "").strip()
            treatment = request.form.get("treatment_plan", "").strip()
            notes = request.form.get("notes", "").strip()
            status = request.form.get("review_status", "")
            follow_up_value = request.form.get("follow_up_date", "").strip()
            if not finding or not treatment or status not in REVIEW_STATUSES:
                flash("Provide a professional finding, treatment recommendation, and valid status.", "danger")
            else:
                try:
                    follow_up_date = date.fromisoformat(follow_up_value) if follow_up_value else None
                except ValueError:
                    flash("Enter a valid follow-up date.", "danger")
                else:
                    if review is None:
                        review = DentistReview(screening_id=screening.id, dentist_id=session["user_id"])
                        db.session.add(review)
                    review.professional_finding = finding
                    review.treatment_plan = treatment
                    review.notes = notes or None
                    review.follow_up_date = follow_up_date
                    review.review_status = status
                    screening.review_status = status
                    if not commit_changes("The dentist review could not be saved. Please try again."):
                        return render_template("dentist/review.html", screening=screening, review=review)
                    flash("Dentist review saved.", "success")
                    return redirect(url_for("dentist_review", screening_id=screening.id))
        return render_template("dentist/review.html", screening=screening, review=review)

    @app.errorhandler(403)
    def forbidden(_error):
        return render_template("error.html", code=403, message="You do not have permission to view this page."), 403

    @app.errorhandler(404)
    def not_found(_error):
        return render_template("error.html", code=404, message="We couldn't find that page or screening."), 404

    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(
            "Unhandled application error: %s",
            error,
            exc_info=(type(error), error, error.__traceback__),
        )
        db.session.rollback()
        return render_template(
            "error.html",
            code=500,
            message="Something went wrong. Please try again in a moment.",
        ), 500

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG", "").lower() == "true")
