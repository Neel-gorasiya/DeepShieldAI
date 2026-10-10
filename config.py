import os
import secrets

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Optional manual .env reader if present (avoiding third-party dependency requirements)
env_path = os.path.join(BASE_DIR, ".env")
if os.path.isfile(env_path):
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip().strip("\"'"))
    except Exception:
        pass

# Security & Session Settings
SECRET_KEY = os.environ.get("SECRET_KEY", secrets.token_hex(32))

# Storage Configuration (Store uploads safely, keeping path configurable)
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", os.path.join(BASE_DIR, "uploads"))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed file extensions
ALLOWED_IMAGE = ["jpg", "jpeg", "png"]
ALLOWED_VIDEO = ["mp4", "avi", "mov"]
ALLOWED_AUDIO = ["wav", "mp3"]
ALLOWED_TEXT = ["txt", "pdf", "docx"]

# Maximum File Upload Size (Default: 50 MB, customizable via MAX_FILE_SIZE_MB env var)
# Prevents memory exhaustion & denial-of-service in ML tensor decoding pipelines
max_mb = int(os.environ.get("MAX_FILE_SIZE_MB", "50"))
MAX_FILE_SIZE = max_mb * 1024 * 1024

# Session Cookie Security Flags
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"
SESSION_COOKIE_SECURE = os.environ.get("SESSION_COOKIE_SECURE", "False").lower() in ("true", "1", "yes")

# Rate Limiting Configuration
UPLOAD_RATE_LIMIT = int(os.environ.get("UPLOAD_RATE_LIMIT", "20"))  # per minute
ACTION_RATE_LIMIT = int(os.environ.get("ACTION_RATE_LIMIT", "40"))  # per minute

# Database Configuration
DATABASE_PATH = os.environ.get("DATABASE_PATH", os.path.join(BASE_DIR, "deepshield.db"))