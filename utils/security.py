import os
import re
import hmac
import secrets
import time
import html
from collections import defaultdict
from threading import Lock

# Magic byte signatures for supported file formats
MAGIC_SIGNATURES = {
    "jpg": [b"\xFF\xD8\xFF"],
    "jpeg": [b"\xFF\xD8\xFF"],
    "png": [b"\x89PNG\r\n\x1a\n"],
    "wav": [b"RIFF"],  # also checked for "WAVE" at offset 8
    "mp3": [b"ID3", b"\xFF\xFB", b"\xFF\xF3", b"\xFF\xF2"],
    "pdf": [b"%PDF-"],
    "docx": [b"PK\x03\x04"],  # Zip archive format used by docx
    "avi": [b"RIFF"],  # also checked for "AVI " at offset 8
}


def validate_file_content(filepath: str, extension: str) -> tuple[bool, str]:
    """
    Validates the actual binary content and magic bytes of an uploaded file.
    Returns (is_valid, error_message).
    """
    if not os.path.exists(filepath):
        return False, "File does not exist."

    file_size = os.path.getsize(filepath)
    if file_size == 0:
        return False, "Uploaded file is empty (0 bytes)."

    ext = extension.lower().lstrip(".")

    try:
        with open(filepath, "rb") as f:
            header = f.read(2048)

        if ext in ["jpg", "jpeg"]:
            if not header.startswith(b"\xFF\xD8\xFF"):
                return False, "Invalid JPEG image header format."

        elif ext == "png":
            if not header.startswith(b"\x89PNG\r\n\x1a\n"):
                return False, "Invalid PNG image header format."

        elif ext == "wav":
            if not (header.startswith(b"RIFF") and b"WAVE" in header[:16]):
                return False, "Invalid WAV audio header format."

        elif ext == "mp3":
            is_mp3 = (
                header.startswith(b"ID3")
                or header.startswith(b"\xFF\xFB")
                or header.startswith(b"\xFF\xF3")
                or header.startswith(b"\xFF\xF2")
            )
            if not is_mp3:
                return False, "Invalid MP3 audio header format."

        elif ext == "pdf":
            if not header.startswith(b"%PDF-"):
                return False, "Invalid PDF document format."

        elif ext == "docx":
            if not header.startswith(b"PK\x03\x04"):
                return False, "Invalid DOCX document container."
            # Verify zip structure contains docx xml components
            import zipfile
            if not zipfile.is_zipfile(filepath):
                return False, "Corrupt or invalid DOCX archive."

        elif ext in ["mp4", "mov"]:
            # MP4/MOV container: check for ftyp, moov, mdat, free, or wide atoms
            valid_atom = any(atom in header[:64] for atom in [b"ftyp", b"moov", b"mdat", b"wide", b"free"])
            if not valid_atom:
                return False, "Invalid MP4/MOV video container format."

        elif ext == "avi":
            if not (header.startswith(b"RIFF") and b"AVI " in header[:16]):
                return False, "Invalid AVI video header format."

        elif ext == "txt":
            # Check for excessive binary null bytes which indicate non-text payloads
            if b"\x00" in header[:512]:
                return False, "Binary file detected where plain text was expected."
            # Verify UTF-8 / ASCII decodability
            try:
                header.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                # Try latin-1 / common text fallback
                try:
                    header.decode("latin-1", errors="strict")
                except Exception:
                    return False, "Text document contains unreadable character encoding."

        return True, ""

    except Exception as e:
        return False, f"File validation error: {str(e)}"


def generate_csrf_token(session: dict) -> str:
    """
    Generates or retrieves a cryptographically secure CSRF token in session.
    """
    if "_csrf_token" not in session:
        session["_csrf_token"] = secrets.token_hex(32)
    return session["_csrf_token"]


def validate_csrf_token(session: dict, submitted_token: str | None) -> bool:
    """
    Validates a submitted CSRF token against the current session token using constant-time comparison.
    """
    if not submitted_token or "_csrf_token" not in session:
        return False
    session_token = session["_csrf_token"]
    return hmac.compare_digest(str(session_token), str(submitted_token))


def sanitize_filename_for_header(filename: str) -> str:
    """
    Sanitizes filenames used in HTTP Content-Disposition headers to prevent CRLF injection
    and header boundary breakouts.
    """
    if not filename:
        return "artifact"
    # Strip path separators, null bytes, quotes, and newlines
    clean = re.sub(r'[\r\n\t"\'\\/]', "_", filename)
    # Allow only safe alphanumeric, underscore, hyphen, and period
    clean = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", clean)
    # Prevent consecutive dots (directory traversal artifacts)
    clean = re.sub(r"\.{2,}", ".", clean).strip("._")
    return clean or "artifact"


def escape_reportlab_text(text: str | None) -> str:
    """
    Safely escapes text for ReportLab Paragraph flowables, preventing XML parsing errors
    and unintended styling injection.
    """
    if text is None:
        return ""
    # Convert to string and escape XML entities
    escaped = html.escape(str(text), quote=True)
    return escaped


class InMemoryRateLimiter:
    """
    Thread-safe in-memory sliding window rate limiter for abuse prevention on sensitive endpoints.
    """
    def __init__(self, requests_per_minute: int = 30):
        self.limit = requests_per_minute
        self.window = 60.0  # seconds
        self.records: dict[str, list[float]] = defaultdict(list)
        self.lock = Lock()

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        with self.lock:
            timestamps = self.records[client_ip]
            # Prune old timestamps
            self.records[client_ip] = [t for t in timestamps if now - t < self.window]
            if len(self.records[client_ip]) >= self.limit:
                return False
            self.records[client_ip].append(now)
            return True


# Global rate limiter instances
upload_limiter = InMemoryRateLimiter(requests_per_minute=20)
action_limiter = InMemoryRateLimiter(requests_per_minute=40)
