"""Comprehensive Test Suite for OralSmart AI.

Covers:
- AI inference with ONNX Runtime
- Missing model handling & graceful fallback
- Corrupted and invalid image rejection
- Flask application factory & database models
- Patient authentication & profile
- Image upload and screening storage
- Screening report rendering
- Dentist review workflow and follow-up scheduling
- Role-based authorization & access control
"""

from io import BytesIO
import json
import os
from pathlib import Path
import pytest
from PIL import Image

import sys
SCRATCH_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRATCH_ROOT / "web"))

from app import create_app, db, User, Screening, DentistReview
from services.ai_service import analyze_dental_image, _model_paths, _unavailable_result


@pytest.fixture
def app():
    """Create test application configured with an in-memory database."""
    test_config = {
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
        "SECRET_KEY": "test-secret-key",
        "UPLOAD_FOLDER": str(SCRATCH_ROOT / "web" / "instance" / "test_uploads"),
        "WTF_CSRF_ENABLED": False,
    }
    app = create_app(test_config)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def dummy_image():
    """Generate an in-memory valid RGB JPEG image."""
    img = Image.new("RGB", (256, 256), color=(220, 180, 140))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


class TestAIService:
    def test_missing_model_graceful_fallback(self, monkeypatch, tmp_path):
        """When ONNX files are missing, returns unavailable status."""
        monkeypatch.setenv("DENTAL_MODEL_PATH", str(tmp_path / "nonexistent.onnx"))
        monkeypatch.setenv("DENTAL_MODEL_LABELS_PATH", str(tmp_path / "nonexistent.json"))

        dummy_path = tmp_path / "test.jpg"
        Image.new("RGB", (100, 100)).save(dummy_path)

        res = analyze_dental_image(str(dummy_path))
        assert res["mode"] == "unavailable"
        assert res["condition"] == "AI model unavailable — no prediction"
        assert res["confidence"] is None

    def test_empty_or_invalid_image_path(self):
        """Empty image path raises ValueError."""
        with pytest.raises(ValueError):
            analyze_dental_image("")

    def test_corrupted_image_handling(self, tmp_path):
        """Corrupted image raises error during processing."""
        corrupt_file = tmp_path / "bad.jpg"
        corrupt_file.write_bytes(b"not an image file content")

        model_path, labels_path = _model_paths()
        if model_path.is_file() and labels_path.is_file():
            with pytest.raises(Exception):
                analyze_dental_image(str(corrupt_file))

    def test_actual_dental_caries_image_inference(self):
        """Run ONNX model on a real intraoral test image and verify outputs."""
        test_dir = SCRATCH_ROOT / "dataset" / "test" / "images"
        if not test_dir.exists():
            pytest.skip("Test images not available")

        images = list(test_dir.glob("*.jpg"))
        if not images:
            pytest.skip("No test images found")

        res = analyze_dental_image(str(images[0]))
        assert res["mode"] == "onnx"
        assert res["condition"] in {
            "No Visible Dental Caries",
            "Dental Caries (Permanent Dentition)",
            "Dental Caries (Primary Dentition)",
        }
        assert 0.0 <= res["confidence"] <= 1.0
        assert "explanation" in res and len(res["explanation"]) > 20
        assert "recommendation" in res and len(res["recommendation"]) > 20
        assert "probabilities" in res
        assert len(res["probabilities"]) == 3


class TestFlaskBackendWorkflows:
    def test_app_initialization(self, app):
        """Verify DB models and tables initialize correctly."""
        with app.app_context():
            assert User.__tablename__ == "users"
            assert Screening.__tablename__ == "screenings"
            assert DentistReview.__tablename__ == "dentist_reviews"

    def test_patient_registration_and_login(self, client):
        """Register patient and verify authenticated session."""
        res = client.post(
            "/register",
            data={
                "name": "Jane Patient",
                "email": "jane@example.com",
                "password": "Password123!",
                "role": "patient",
                "phone": "+1234567890",
            },
            follow_redirects=True,
        )
        assert res.status_code == 200
        assert b"Jane Patient" in res.data

        # Logout and Login
        client.post("/logout", follow_redirects=True)
        login_res = client.post(
            "/login",
            data={"email": "jane@example.com", "password": "Password123!"},
            follow_redirects=True,
        )
        assert login_res.status_code == 200
        assert b"Jane Patient" in login_res.data

    def test_dentist_registration_and_dashboard(self, client):
        """Register dentist and verify access to dentist dashboard."""
        client.post(
            "/register",
            data={
                "name": "Dr. Sarah Dentist",
                "email": "drsarah@example.com",
                "password": "Password123!",
                "role": "dentist",
                "phone": "+1987654321",
            },
            follow_redirects=True,
        )
        dashboard_res = client.get("/dentist")
        assert dashboard_res.status_code == 200
        assert b"Dr. Sarah Dentist" in dashboard_res.data

    def test_patient_upload_and_screening_flow(self, client, dummy_image):
        """Patient uploads image -> AI prediction recorded -> Report rendered."""
        # Register and log in patient
        client.post(
            "/register",
            data={
                "name": "Alex Patient",
                "email": "alex@example.com",
                "password": "Password123!",
                "role": "patient",
            },
            follow_redirects=True,
        )

        upload_res = client.post(
            "/patient/upload",
            data={"image": (dummy_image, "dental_test.jpg")},
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        assert upload_res.status_code == 200
        assert b"AI Preliminary Screening" in upload_res.data or b"SCREENING REPORT" in upload_res.data

        # Verify Screening record in DB
        with client.application.app_context():
            screening = db.session.scalar(db.select(Screening).filter_by(patient_id=1))
            assert screening is not None
            assert screening.ai_result is not None
            assert screening.model_mode in {"onnx", "unavailable"}

        # Check screening history
        hist_res = client.get("/patient/history")
        assert hist_res.status_code == 200
        assert b"No Visible Dental Caries" in hist_res.data
        assert b"/patient/screenings/1" in hist_res.data

    def test_dentist_review_workflow(self, client, dummy_image):
        """Complete workflow: Patient requests review -> Dentist enters finding & follow-up."""
        # 1. Register Patient and upload
        client.post(
            "/register",
            data={
                "name": "Bob Patient",
                "email": "bob@example.com",
                "password": "Password123!",
                "role": "patient",
            },
            follow_redirects=True,
        )
        client.post(
            "/patient/upload",
            data={"image": (dummy_image, "bob_tooth.jpg")},
            content_type="multipart/form-data",
            follow_redirects=True,
        )

        # 2. Patient requests dentist review
        req_res = client.post("/patient/screenings/1/request-review", follow_redirects=True)
        assert req_res.status_code == 200
        assert b"review queue" in req_res.data.lower()

        # 3. Register Dentist and log in
        client.post("/logout", follow_redirects=True)
        client.post(
            "/register",
            data={
                "name": "Dr. Smith",
                "email": "drsmith@example.com",
                "password": "Password123!",
                "role": "dentist",
            },
            follow_redirects=True,
        )

        # 4. Dentist reviews screening #1
        review_post = client.post(
            "/dentist/screenings/1",
            data={
                "professional_finding": "Mild enamel demineralization on upper right first molar.",
                "treatment_plan": "Fluoride varnish application and re-evaluation in 3 months.",
                "notes": "Recommend soft bristle toothbrush.",
                "review_status": "Reviewed",
                "follow_up_date": "2026-11-20",
            },
            follow_redirects=True,
        )
        assert review_post.status_code == 200
        assert b"Dentist review saved" in review_post.data

        # 5. Patient logs back in and sees completed review
        client.post("/logout", follow_redirects=True)
        client.post("/login", data={"email": "bob@example.com", "password": "Password123!"}, follow_redirects=True)
        report_res = client.get("/patient/screenings/1")
        assert b"DENTIST REVIEW" in report_res.data
        assert b"Fluoride varnish application" in report_res.data

    def test_access_control_restrictions(self, client, dummy_image):
        """Ensure patient cannot access dentist reviews or other patients' records."""
        # Patient 1
        client.post(
            "/register",
            data={"name": "P1", "email": "p1@example.com", "password": "Password123!", "role": "patient"},
            follow_redirects=True,
        )
        client.post(
            "/patient/upload",
            data={"image": (dummy_image, "p1.jpg")},
            content_type="multipart/form-data",
            follow_redirects=True,
        )

        # Patient attempts to access dentist dashboard
        den_access = client.get("/dentist")
        assert den_access.status_code == 403

        # Patient 2 logs in
        client.post("/logout", follow_redirects=True)
        client.post(
            "/register",
            data={"name": "P2", "email": "p2@example.com", "password": "Password123!", "role": "patient"},
            follow_redirects=True,
        )

        # Patient 2 tries to view Patient 1's screening (#1)
        p1_record = client.get("/patient/screenings/1")
        assert p1_record.status_code == 404

    def test_invalid_file_upload_rejected(self, client):
        """Reject non-image extensions (.txt, .pdf)."""
        client.post(
            "/register",
            data={"name": "Test User", "email": "testupload@example.com", "password": "Password123!", "role": "patient"},
            follow_redirects=True,
        )
        fake_file = BytesIO(b"this is plain text, not an image")
        res = client.post(
            "/patient/upload",
            data={"image": (fake_file, "notes.txt")},
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        assert b"Upload a JPG, JPEG, or PNG image" in res.data

    def test_unauthenticated_routes_redirect(self, client):
        """Ensure unauthenticated access redirects to login."""
        res_patient = client.get("/patient", follow_redirects=False)
        assert res_patient.status_code == 302
        assert "/login" in res_patient.headers["Location"]

        res_dentist = client.get("/dentist", follow_redirects=False)
        assert res_dentist.status_code == 302
        assert "/login" in res_dentist.headers["Location"]

