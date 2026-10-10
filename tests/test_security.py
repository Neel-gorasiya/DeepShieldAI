import io
import os
import unittest
import tempfile
import json
import sqlite3

from app import app
import database
from utils.security import (
    validate_file_content,
    generate_csrf_token,
    validate_csrf_token,
    sanitize_filename_for_header,
    escape_reportlab_text,
    upload_limiter,
    action_limiter,
)
from utils.pdf_generator import generate_pdf_report


class SecurityTestSuite(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = True
        self.client = app.test_client()

        # Create an isolated temporary test database to protect the production DB
        self.test_db_fd, self.test_db_path = tempfile.mkstemp(suffix=".db")
        database.DATABASE_PATH = self.test_db_path
        database.init_db()

    def tearDown(self):
        os.close(self.test_db_fd)
        try:
            if os.path.exists(self.test_db_path):
                os.remove(self.test_db_path)
            # Remove any SQLite WAL / SHM files if present
            for ext in ["-wal", "-shm", "-journal"]:
                if os.path.exists(self.test_db_path + ext):
                    os.remove(self.test_db_path + ext)
        except Exception:
            pass

    # =========================================================================
    # 1. SECURITY HEADERS & COOKIES TESTS
    # =========================================================================
    def test_security_headers_present(self):
        """Verify that all critical security headers are set on HTTP responses."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

        headers = response.headers
        self.assertIn("Content-Security-Policy", headers)
        self.assertIn("X-Content-Type-Options", headers)
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertIn("X-Frame-Options", headers)
        self.assertEqual(headers["X-Frame-Options"], "SAMEORIGIN")
        self.assertIn("Referrer-Policy", headers)
        self.assertEqual(headers["Referrer-Policy"], "strict-origin-when-cross-origin")
        self.assertIn("Permissions-Policy", headers)

    def test_session_cookie_settings(self):
        """Verify that session cookies have HttpOnly and SameSite configured."""
        self.assertTrue(app.config["SESSION_COOKIE_HTTPONLY"])
        self.assertEqual(app.config["SESSION_COOKIE_SAMESITE"], "Lax")

    # =========================================================================
    # 2. CSRF PROTECTION TESTS
    # =========================================================================
    def test_predict_rejected_without_csrf_token(self):
        """Verify POST /predict returns 400 Bad Request when CSRF token is missing."""
        data = {
            "file": (io.BytesIO(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"), "test.jpg"),
        }
        response = self.client.post("/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Bad Request", response.data)

    def test_predict_rejected_with_invalid_csrf_token(self):
        """Verify POST /predict returns 400 when an invalid CSRF token is submitted."""
        with self.client.session_transaction() as sess:
            sess["_csrf_token"] = "valid_token_12345"

        data = {
            "csrf_token": "forged_invalid_token",
            "file": (io.BytesIO(b"\xFF\xD8\xFF\xE0\x00\x10JFIF"), "test.jpg"),
        }
        response = self.client.post("/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)

    def test_delete_rejected_without_csrf_token(self):
        """Verify POST /delete/<id> returns 400 when CSRF token is missing."""
        response = self.client.post("/delete/1")
        self.assertEqual(response.status_code, 400)

    def test_delete_succeeds_with_valid_csrf_token(self):
        """Verify POST /delete/<id> succeeds with valid CSRF token and deletes the record."""
        # Insert a test record
        record_id = database.save_verification(
            filename="delete_target.jpg",
            media_type="image",
            result={"prediction": "Human Created Image", "ai_score": 10.0, "human_score": 90.0, "reason": "Test"},
        )

        with self.client.session_transaction() as sess:
            sess["_csrf_token"] = "test_csrf_token_secret"

        response = self.client.post(
            f"/delete/{record_id}",
            data={"csrf_token": "test_csrf_token_secret"},
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)

        # Confirm deleted from DB
        deleted_record = database.get_verification(record_id)
        self.assertIsNone(deleted_record)

    # =========================================================================
    # 3. FILE UPLOAD & MAGIC BYTE VALIDATION TESTS
    # =========================================================================
    def test_empty_file_rejected(self):
        """Verify that an empty (0 bytes) upload is rejected."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name

        try:
            is_valid, msg = validate_file_content(temp_path, "png")
            self.assertFalse(is_valid)
            self.assertIn("empty", msg.lower())
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_mismatched_binary_payload_rejected(self):
        """Verify that executable/shell payload renamed to .jpg is rejected by header inspection."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff\x00\x00")
            temp_path = f.name

        try:
            is_valid, msg = validate_file_content(temp_path, "jpg")
            self.assertFalse(is_valid)
            self.assertIn("invalid jpeg", msg.lower())
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_valid_image_headers(self):
        """Verify that genuine JPEG and PNG header signatures pass validation."""
        # JPEG
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01")
            temp_jpg = f.name

        # PNG
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
            temp_png = f.name

        try:
            valid_jpg, _ = validate_file_content(temp_jpg, "jpg")
            self.assertTrue(valid_jpg)

            valid_png, _ = validate_file_content(temp_png, "png")
            self.assertTrue(valid_png)
        finally:
            if os.path.exists(temp_jpg):
                os.remove(temp_jpg)
            if os.path.exists(temp_png):
                os.remove(temp_png)

    def test_pdf_magic_bytes(self):
        """Verify PDF header signature validation."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.7\n%raw content")
            temp_pdf = f.name

        try:
            valid_pdf, _ = validate_file_content(temp_pdf, "pdf")
            self.assertTrue(valid_pdf)
        finally:
            if os.path.exists(temp_pdf):
                os.remove(temp_pdf)

    # =========================================================================
    # 4. HTTP HEADER INJECTION & PDF REPORT ESCAPING
    # =========================================================================
    def test_header_filename_sanitization(self):
        """Verify CRLF injection and quotes are stripped from Content-Disposition headers."""
        malicious_name = 'report\r\nSet-Cookie: evil=1\r\n"filename=pwned.pdf'
        clean = sanitize_filename_for_header(malicious_name)
        self.assertNotIn("\r", clean)
        self.assertNotIn("\n", clean)
        self.assertNotIn('"', clean)
        self.assertNotIn("'", clean)

    def test_reportlab_xml_escaping(self):
        """Verify that ReportLab dynamic text is safely escaped to prevent XML parse crashes."""
        malicious_input = '<font color="red"><b>XSS Test</b></font> & "quotes" <script>'
        escaped = escape_reportlab_text(malicious_input)
        self.assertIn("&lt;font", escaped)
        self.assertIn("&amp;", escaped)
        self.assertNotIn("<script>", escaped)

    def test_pdf_generation_with_malicious_metadata(self):
        """Verify PDF generation succeeds without crash when record contains malicious characters."""
        malicious_record = {
            "id": 42,
            "filename": '<script>alert("xss")</script>.png',
            "media_type": "image",
            "prediction": 'AI Generated <font color="red">Image</font>',
            "verdict_category": "ai",
            "ai_score": 95.5,
            "human_score": 4.5,
            "reason": 'Special & symbols < > " \' in reason text',
            "file_size": 2048,
            "created_at": "2026-10-10 10:00:00",
            "sha256": "abcdef1234567890",
            "model_name": "CLIP <ViT-B/32> & neural probe",
            "processing_time": "0.15s",
            "metadata_json": json.dumps({"details": {"tag": "<b>malicious</b>", "amp": "1 & 2"}}),
        }

        # Should generate valid PDF bytes without ReportLab XML syntax exception
        pdf_bytes = generate_pdf_report(malicious_record)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))

    # =========================================================================
    # 5. DATABASE SECURITY & SQL INJECTION TESTS
    # =========================================================================
    def test_history_search_sql_injection(self):
        """Verify search queries with SQL injection payloads do not break or alter query structure."""
        database.save_verification(
            filename="sample_target.jpg",
            media_type="image",
            result={"prediction": "AI Generated Image", "ai_score": 90.0, "human_score": 10.0, "reason": "Sample"},
        )

        sql_payloads = [
            "' OR '1'='1",
            "'; DROP TABLE verifications; --",
            "1 UNION SELECT 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17 --",
        ]

        for payload in sql_payloads:
            records = database.get_history_records(search=payload)
            self.assertIsInstance(records, list)

        # Confirm table is intact
        with database.get_db_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM verifications").fetchone()[0]
            self.assertGreaterEqual(count, 1)

    def test_history_sort_whitelist(self):
        """Verify sort parameter only accepts valid whitelisted sort orders."""
        records = database.get_history_records(sort_by="; DROP TABLE verifications; --")
        self.assertIsInstance(records, list)

    def test_database_wal_mode_enabled(self):
        """Verify WAL mode is enabled for database concurrency and resilience."""
        with database.get_db_connection() as conn:
            journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
            self.assertEqual(journal_mode.lower(), "wal")

    # =========================================================================
    # 6. RATE LIMITER UNIT TESTS
    # =========================================================================
    def test_rate_limiter_blocks_excess_requests(self):
        """Verify that rate limiter blocks requests exceeding the threshold."""
        from utils.security import InMemoryRateLimiter
        limiter = InMemoryRateLimiter(requests_per_minute=3)
        client_ip = "192.168.1.100"

        # First 3 requests should be allowed
        self.assertTrue(limiter.is_allowed(client_ip))
        self.assertTrue(limiter.is_allowed(client_ip))
        self.assertTrue(limiter.is_allowed(client_ip))

        # 4th request must be rejected
        self.assertFalse(limiter.is_allowed(client_ip))

    # =========================================================================
    # 7. ERROR HANDLER TESTS
    # =========================================================================
    def test_404_error_page(self):
        """Verify 404 handler returns sanitized friendly page without stack traces."""
        response = self.client.get("/result/99999999")
        self.assertEqual(response.status_code, 404)
        self.assertIn(b"Record Not Found", response.data)
        self.assertNotIn(b"Traceback", response.data)


if __name__ == "__main__":
    unittest.main()
