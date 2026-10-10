import warnings
warnings.filterwarnings("ignore")

import os
import time
import json
import uuid
import logging

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

logger = logging.getLogger("deepshield.app")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    make_response,
    session,
    abort,
)
from werkzeug.utils import secure_filename

from config import (
    SECRET_KEY,
    UPLOAD_FOLDER,
    ALLOWED_IMAGE,
    ALLOWED_VIDEO,
    ALLOWED_AUDIO,
    ALLOWED_TEXT,
    MAX_FILE_SIZE,
    SESSION_COOKIE_HTTPONLY,
    SESSION_COOKIE_SAMESITE,
    SESSION_COOKIE_SECURE,
)
from models.image_detector import detect_image
from models.audio_detector import detect_audio
from models.video_detector import detect_video
from models.text_detector import detect_text
import database
from utils.forensics import extract_media_forensics
from utils.pdf_generator import generate_pdf_report
from utils.security import (
    validate_file_content,
    generate_csrf_token,
    validate_csrf_token,
    sanitize_filename_for_header,
    upload_limiter,
    action_limiter,
)

app = Flask(__name__)
app.secret_key = SECRET_KEY

# Secure session cookies
app.config["SESSION_COOKIE_HTTPONLY"] = SESSION_COOKIE_HTTPONLY
app.config["SESSION_COOKIE_SAMESITE"] = SESSION_COOKIE_SAMESITE
app.config["SESSION_COOKIE_SECURE"] = SESSION_COOKIE_SECURE

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize persistent SQLite database
database.init_db()

MODEL_NAMES = {
    "image": "CLIP ViT-B/32 Zero-Shot Classifier",
    "video": "EfficientNet-B0 + High-Frequency Texture Forensics",
    "audio": "Wav2Vec2 + Acoustic F0 & Shimmer Signal Analysis",
    "text": "RoBERTa-Base OpenAI Neural Token Detector",
    "other": "General File Header Inspection Pipeline",
}

EMPTY_RESULT = {
    "prediction": "Unsupported File",
    "ai_score": 0.0,
    "human_score": 0.0,
    "reason": "The uploaded file format or content structure is not supported by the multi-modal forensic pipeline.",
}


@app.context_processor
def inject_csrf_token():
    """
    Exposes a callable csrf_token() helper to Jinja templates.
    """
    return dict(csrf_token=lambda: generate_csrf_token(session))


@app.after_request
def apply_security_headers(response):
    """
    Applies comprehensive security headers to prevent XSS, clickjacking, MIME sniffing,
    and unauthorized resource framing.
    """
    csp_policy = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data: blob:; "
        "media-src 'self' data: blob:; "
        "frame-ancestors 'none'; "
        "object-src 'none'; "
        "base-uri 'self';"
    )
    response.headers["Content-Security-Policy"] = csp_policy
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), camera=(), microphone=()"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response


def get_client_ip() -> str:
    """Safely gets client IP for rate limiting."""
    if request.headers.get("X-Forwarded-For"):
        return request.headers.get("X-Forwarded-For").split(",")[0].strip()
    return request.remote_addr or "127.0.0.1"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    stats = database.get_dashboard_data()
    return render_template("dashboard.html", data=stats)


@app.route("/history")
def history():
    search = request.args.get("search", "").strip()
    media_filter = request.args.get("media", "all").strip().lower()
    verdict_filter = request.args.get("verdict", "all").strip().lower()
    date_filter = request.args.get("date", "").strip()
    sort_by = request.args.get("sort", "newest").strip().lower()

    records = database.get_history_records(
        search=search,
        media_filter=media_filter,
        verdict_filter=verdict_filter,
        date_filter=date_filter,
        sort_by=sort_by,
    )
    return render_template(
        "history.html",
        records=records,
        search=search,
        media_filter=media_filter,
        verdict_filter=verdict_filter,
        date_filter=date_filter,
        sort_by=sort_by,
    )


@app.route("/predict", methods=["POST"])
def predict():
    # 1. Rate Limiting Check
    client_ip = get_client_ip()
    if not upload_limiter.is_allowed(client_ip):
        abort(429)

    # 2. CSRF Token Validation
    token = request.form.get("csrf_token")
    if not validate_csrf_token(session, token):
        abort(400)

    # 3. File Retrieval & Basic Name Sanitization
    file = request.files.get("file")
    if file is None or file.filename == "":
        err_result = {
            "prediction": "No File Uploaded",
            "ai_score": 0.0,
            "human_score": 0.0,
            "reason": "Please choose a file before submitting for analysis.",
        }
        return render_template("result.html", result=err_result, file="", record=None)

    raw_filename = file.filename
    display_filename = secure_filename(raw_filename)
    if not display_filename or "." not in display_filename:
        display_filename = "unnamed_artifact.dat"

    extension = display_filename.rsplit(".", 1)[-1].lower()
    allowed = ALLOWED_IMAGE + ALLOWED_VIDEO + ALLOWED_AUDIO + ALLOWED_TEXT

    # 4. Check Allowed Extension
    if extension not in allowed:
        record_id = database.save_verification(
            filename=display_filename,
            media_type="other",
            result=EMPTY_RESULT,
            file_size=0,
            status="warning",
            model_name=MODEL_NAMES["other"],
            processing_time="0.01s",
        )
        return redirect(url_for("view_result", record_id=record_id))

    # 5. Generate safe unique random storage filename (prevents file overwrite & traversal)
    storage_filename = f"{uuid.uuid4().hex}.{extension}"
    storage_path = os.path.join(app.config["UPLOAD_FOLDER"], storage_filename)

    try:
        file.save(storage_path)
    except Exception as e:
        logger.error(f"Failed to save upload to disk: {e}")
        abort(500)

    file_size = os.path.getsize(storage_path) if os.path.exists(storage_path) else 0

    # 6. Deep Binary & Magic-Byte File Validation
    is_valid_format, validation_error = validate_file_content(storage_path, extension)
    if not is_valid_format:
        # Clean up invalid payload
        try:
            if os.path.exists(storage_path):
                os.remove(storage_path)
        except Exception:
            pass

        invalid_result = {
            "prediction": "Corrupted or Invalid File",
            "ai_score": 0.0,
            "human_score": 0.0,
            "reason": f"File validation failed: {validation_error}",
        }
        record_id = database.save_verification(
            filename=display_filename,
            media_type="other",
            result=invalid_result,
            file_size=file_size,
            status="warning",
            model_name=MODEL_NAMES["other"],
            processing_time="0.01s",
        )
        return redirect(url_for("view_result", record_id=record_id))

    # 7. Modality Classification
    media_type = "other"
    if extension in ALLOWED_IMAGE:
        media_type = "image"
    elif extension in ALLOWED_VIDEO:
        media_type = "video"
    elif extension in ALLOWED_AUDIO:
        media_type = "audio"
    elif extension in ALLOWED_TEXT:
        media_type = "text"

    model_name = MODEL_NAMES.get(media_type, MODEL_NAMES["other"])
    status = "completed"
    error_details = None

    # 8. Measure Inference Execution Time & Run Model
    start_time = time.time()
    try:
        if media_type == "image":
            result = detect_image(storage_path)
        elif media_type == "video":
            result = detect_video(storage_path)
        elif media_type == "audio":
            result = detect_audio(storage_path)
        elif media_type == "text":
            result = detect_text(storage_path)
        else:
            result = EMPTY_RESULT
            status = "warning"

        if result.get("prediction") == "Error":
            status = "failed"
            error_details = result.get("reason")
    except Exception as e:
        status = "failed"
        error_details = "Inference execution error occurred."
        logger.error(f"Inference exception on {display_filename}: {e}")
        result = {"prediction": "Error", "ai_score": 0.0, "human_score": 0.0, "reason": "Processing encountered an unexpected error."}

    elapsed_time = f"{round(time.time() - start_time, 2)}s"

    # 9. Extract Forensic Metadata (SHA-256, format details)
    forensic_metadata = extract_media_forensics(storage_path, media_type)
    sha256_hash = forensic_metadata.get("sha256", "N/A")

    # Clean up temporary upload file after inference to preserve storage and privacy
    try:
        if os.path.exists(storage_path):
            os.remove(storage_path)
    except Exception as e:
        logger.warning(f"Could not remove temporary payload file {storage_path}: {e}")

    # 10. Save completed analysis into persistent SQLite database
    record_id = database.save_verification(
        filename=display_filename,
        media_type=media_type,
        result=result,
        file_size=file_size,
        status=status,
        error_details=error_details,
        sha256=sha256_hash,
        model_name=model_name,
        processing_time=elapsed_time,
        metadata=forensic_metadata,
    )

    # Post/Redirect/Get pattern prevents duplicate records on page refresh
    return redirect(url_for("view_result", record_id=record_id))


@app.route("/result/<int:record_id>")
def view_result(record_id):
    if record_id <= 0:
        abort(404)

    record = database.get_verification(record_id)
    if not record:
        abort(404)

    # Parse metadata safely
    metadata = {}
    if record.get("metadata_json"):
        try:
            metadata = json.loads(record["metadata_json"])
        except Exception:
            metadata = {}

    result_dict = {
        "prediction": record["prediction"],
        "ai_score": record["ai_score"],
        "human_score": record["human_score"],
        "reason": record["reason"],
    }
    return render_template(
        "result.html",
        result=result_dict,
        file=record["filename"],
        record=record,
        metadata=metadata,
    )


@app.route("/result/<int:record_id>/pdf")
def download_pdf(record_id):
    if record_id <= 0:
        abort(404)

    record = database.get_verification(record_id)
    if not record:
        abort(404)

    pdf_bytes = generate_pdf_report(record)
    response = make_response(pdf_bytes)
    response.headers["Content-Type"] = "application/pdf"
    response.headers["X-Content-Type-Options"] = "nosniff"

    raw_filename = record.get("filename", "report")
    safe_base_name = sanitize_filename_for_header(raw_filename)
    attachment_name = f"DeepShield_Report_#{record_id:04d}_{safe_base_name}.pdf"
    response.headers["Content-Disposition"] = f'attachment; filename="{attachment_name}"'
    return response


@app.route("/delete/<int:record_id>", methods=["POST"])
def delete_record(record_id):
    # Rate Limiting & CSRF Validation
    client_ip = get_client_ip()
    if not action_limiter.is_allowed(client_ip):
        abort(429)

    token = request.form.get("csrf_token")
    if not validate_csrf_token(session, token):
        abort(400)

    if record_id <= 0:
        abort(404)

    success = database.delete_verification(record_id)
    if success:
        flash(f"Verification record #{record_id:04d} was successfully deleted.", "success")
    else:
        flash("Failed to delete record.", "error")
    return redirect(url_for("history"))


# Production-Ready Safe Error Handlers
@app.errorhandler(400)
def bad_request(e):
    return render_template(
        "result.html",
        result={
            "prediction": "Bad Request",
            "ai_score": 0.0,
            "human_score": 0.0,
            "reason": "The request could not be verified or CSRF validation token was invalid.",
        },
        file="",
        record=None,
        metadata={},
    ), 400


@app.errorhandler(404)
def not_found(e):
    return render_template(
        "result.html",
        result={
            "prediction": "Record Not Found",
            "ai_score": 0.0,
            "human_score": 0.0,
            "reason": "The requested verification audit record or page does not exist.",
        },
        file="",
        record=None,
        metadata={},
    ), 404


@app.errorhandler(413)
def file_too_large(e):
    return render_template(
        "result.html",
        result={
            "prediction": "File Too Large",
            "ai_score": 0.0,
            "human_score": 0.0,
            "reason": f"Upload exceeds maximum server payload size ({MAX_FILE_SIZE // (1024 * 1024)} MB).",
        },
        file="",
        record=None,
        metadata={},
    ), 413


@app.errorhandler(429)
def too_many_requests(e):
    return render_template(
        "result.html",
        result={
            "prediction": "Rate Limit Exceeded",
            "ai_score": 0.0,
            "human_score": 0.0,
            "reason": "Too many requests submitted. Please wait a minute before submitting additional verifications.",
        },
        file="",
        record=None,
        metadata={},
    ), 429


@app.errorhandler(500)
def server_error(e):
    logger.error(f"Internal 500 error: {e}")
    return render_template(
        "result.html",
        result={
            "prediction": "Internal Server Error",
            "ai_score": 0.0,
            "human_score": 0.0,
            "reason": "An unexpected server error occurred during processing. Please try again later.",
        },
        file="",
        record=None,
        metadata={},
    ), 500


if __name__ == "__main__":
    app.run(debug=False)
