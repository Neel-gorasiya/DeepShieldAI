import sqlite3
import os
import json
import logging
from datetime import datetime
from config import DATABASE_PATH

logger = logging.getLogger("deepshield.database")


def get_db_connection():
    """
    Creates a thread-safe SQLite connection configured with WAL mode,
    busy timeout, and row factories for high concurrency and resilience.
    """
    conn = sqlite3.connect(DATABASE_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    # Apply concurrency and resilience PRAGMAs
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
    except Exception as e:
        logger.warning(f"Error applying PRAGMAs: {e}")
    return conn


def init_db():
    """
    Initializes the verification database schema safely with non-destructive migrations.
    """
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS verifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                media_type TEXT NOT NULL,
                prediction TEXT NOT NULL,
                verdict_category TEXT NOT NULL,
                ai_score REAL NOT NULL DEFAULT 0.0,
                human_score REAL NOT NULL DEFAULT 0.0,
                confidence REAL NOT NULL DEFAULT 0.0,
                reason TEXT,
                file_size INTEGER DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'completed',
                error_details TEXT,
                sha256 TEXT,
                model_name TEXT,
                processing_time TEXT,
                metadata_json TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        # Automatic non-destructive column migrations with validated identifiers
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(verifications)")
        existing_cols = {row["name"] for row in cursor.fetchall()}

        columns_to_add = {
            "status": "TEXT NOT NULL DEFAULT 'completed'",
            "error_details": "TEXT",
            "sha256": "TEXT",
            "model_name": "TEXT",
            "processing_time": "TEXT",
            "metadata_json": "TEXT",
        }

        # Safe whitelist validation of column definitions
        for col_name, col_def in columns_to_add.items():
            if col_name not in existing_cols and col_name.isidentifier():
                conn.execute(f"ALTER TABLE verifications ADD COLUMN {col_name} {col_def}")

        conn.commit()


def determine_category_and_confidence(prediction, ai_score, human_score):
    p = (prediction or "").strip()
    ai_candidates = [
        "AI Generated Image", "AI Generated Video", "AI Generated Voice", "AI Generated Text"
    ]
    human_candidates = [
        "Human Created Image", "Human Recorded Video", "Human Voice", "Human Written Text"
    ]

    is_ai = (p in ai_candidates) or ("AI Generated" in p) or ("Synthetic" in p) or (p == "Fake")
    is_human = (p in human_candidates) or ("Human" in p) or ("Authentic" in p) or (p == "Real")

    if is_ai:
        category = "ai"
        confidence = float(ai_score if ai_score is not None else 0.0)
    elif is_human:
        category = "human"
        confidence = float(human_score if human_score is not None else 0.0)
    else:
        category = "alert"
        confidence = 0.0

    return category, round(confidence, 2)


def save_verification(
    filename,
    media_type,
    result,
    file_size=0,
    status="completed",
    error_details=None,
    sha256="N/A",
    model_name="Neural Classifier",
    processing_time="0.0s",
    metadata=None,
):
    init_db()
    prediction = result.get("prediction", "Unknown") if isinstance(result, dict) else "Unknown"
    ai_score = float(result.get("ai_score", 0.0) or 0.0) if isinstance(result, dict) else 0.0
    human_score = float(result.get("human_score", 0.0) or 0.0) if isinstance(result, dict) else 0.0
    reason = result.get("reason", "") if isinstance(result, dict) else ""
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    category, confidence = determine_category_and_confidence(prediction, ai_score, human_score)

    if prediction in ["Error", "Unsupported File", "File Too Large", "No File Uploaded"]:
        if status == "completed":
            status = "failed" if prediction == "Error" else "warning"

    metadata_str = json.dumps(metadata) if metadata else "{}"

    # Sanitize inputs
    sanitized_filename = str(filename)[:255] if filename else "unnamed_artifact"
    sanitized_media = str(media_type).lower()[:32] if media_type else "other"

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO verifications (
                filename, media_type, prediction, verdict_category,
                ai_score, human_score, confidence, reason, file_size,
                status, error_details, sha256, model_name, processing_time,
                metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sanitized_filename,
                sanitized_media,
                str(prediction)[:128],
                str(category)[:32],
                round(ai_score, 2),
                round(human_score, 2),
                round(confidence, 2),
                str(reason)[:2000],
                max(0, int(file_size)),
                str(status)[:32],
                str(error_details)[:1000] if error_details else None,
                str(sha256)[:128],
                str(model_name)[:128],
                str(processing_time)[:32],
                metadata_str,
                created_at,
            ),
        )
        conn.commit()
        return cursor.lastrowid


def get_verification(record_id):
    """
    Safely retrieves a single verification record by integer ID.
    """
    if not isinstance(record_id, int) or record_id <= 0:
        try:
            record_id = int(record_id)
            if record_id <= 0:
                return None
        except (ValueError, TypeError):
            return None

    init_db()
    try:
        with get_db_connection() as conn:
            row = conn.execute("SELECT * FROM verifications WHERE id = ?", (record_id,)).fetchone()
            if row:
                return dict(row)
            return None
    except Exception as e:
        logger.error(f"Error querying verification record #{record_id}: {e}")
        return None


def delete_verification(record_id):
    """
    Safely deletes a verification record by integer ID.
    """
    if not isinstance(record_id, int) or record_id <= 0:
        try:
            record_id = int(record_id)
            if record_id <= 0:
                return False
        except (ValueError, TypeError):
            return False

    init_db()
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM verifications WHERE id = ?", (record_id,))
            conn.commit()
            return cursor.rowcount > 0
    except Exception as e:
        logger.error(f"Error deleting verification record #{record_id}: {e}")
        return False


def get_dashboard_data():
    """
    Calculates summary statistics and aggregated metrics across records.
    """
    init_db()
    with get_db_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM verifications").fetchone()[0]
        human_count = conn.execute(
            "SELECT COUNT(*) FROM verifications WHERE verdict_category = 'human'"
        ).fetchone()[0]
        ai_count = conn.execute(
            "SELECT COUNT(*) FROM verifications WHERE verdict_category = 'ai'"
        ).fetchone()[0]
        alert_count = conn.execute(
            "SELECT COUNT(*) FROM verifications WHERE verdict_category = 'alert'"
        ).fetchone()[0]

        # Media breakdown
        media_types = {"image": 0, "video": 0, "audio": 0, "text": 0}
        type_rows = conn.execute(
            "SELECT media_type, COUNT(*) as count FROM verifications GROUP BY media_type"
        ).fetchall()
        for r in type_rows:
            m_type = (r["media_type"] or "").lower()
            if m_type in media_types:
                media_types[m_type] = r["count"]
            else:
                media_types[m_type] = r["count"]

        # Recent records (latest 8)
        recent_rows = conn.execute(
            "SELECT * FROM verifications ORDER BY id DESC LIMIT 8"
        ).fetchall()
        recent_activity = [dict(r) for r in recent_rows]

        return {
            "total": total,
            "human_count": human_count,
            "ai_count": ai_count,
            "alert_count": alert_count,
            "media_types": media_types,
            "recent_activity": recent_activity,
        }


def get_history_records(search="", media_filter="all", verdict_filter="all", date_filter="", sort_by="newest"):
    """
    Retrieves filtered history records using parameterized queries and strict whitelist sorting.
    """
    init_db()
    query = "SELECT * FROM verifications WHERE 1=1"
    params = []

    if search:
        search_term = str(search).strip()[:100]
        query += " AND (filename LIKE ? OR prediction LIKE ? OR reason LIKE ? OR sha256 LIKE ?)"
        term = f"%{search_term}%"
        params.extend([term, term, term, term])

    if media_filter and str(media_filter).lower() != "all":
        query += " AND media_type = ?"
        params.append(str(media_filter).lower().strip()[:32])

    if verdict_filter and str(verdict_filter).lower() != "all":
        query += " AND verdict_category = ?"
        params.append(str(verdict_filter).lower().strip()[:32])

    if date_filter:
        clean_date = str(date_filter).strip()[:10]
        query += " AND created_at LIKE ?"
        params.append(f"{clean_date}%")

    # Strict whitelist for ORDER BY clause preventing SQL structure tampering
    order_clauses = {
        "oldest": " ORDER BY id ASC",
        "confidence_desc": " ORDER BY confidence DESC, id DESC",
        "confidence_asc": " ORDER BY confidence ASC, id DESC",
        "filename_asc": " ORDER BY filename COLLATE NOCASE ASC",
        "newest": " ORDER BY id DESC",
    }
    query += order_clauses.get(str(sort_by).lower(), " ORDER BY id DESC")

    with get_db_connection() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]
