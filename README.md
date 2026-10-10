# DeepShield AI — AI-Powered Deepfake Detection & Media Verification

**DeepShield AI** is a multi-modal forensic intelligence and deepfake detection platform developed as a software engineering final-year project. It inspects uploaded media payloads — images, videos, audio recordings, or documents — to estimate whether they are **AI-generated / synthetic / deepfakes** or **authentic human creations**.

The system provides dual AI/Human confidence probabilities, SHA-256 cryptographic artifact checksums, detailed forensic signal explanations, execution runtime metrics, a persistent SQLite audit trail, an interactive dashboard, and downloadable PDF verification certificates.

---

## Capabilities & Modalities

| Modality | Accepted Formats | Underlying Architecture & Techniques |
|---|---|---|
| **Image** | `.jpg`, `.jpeg`, `.png` | CLIP ViT-B/32 zero-shot semantic prompt vector matching (`"real photo"` vs `"ai generated image"`) + Dimension & Color-space metadata |
| **Video** | `.mp4`, `.avi`, `.mov` | EfficientNet-B0 visual representations + Laplacian & DCT high-frequency texture artifact heuristics + Frame/Duration metrics |
| **Audio** | `.wav`, `.mp3` | Acoustic artifact signal processing (pitch F0 stability, shimmer/energy variance, high-frequency spectral detail) + Wav2Vec2 embedding variance |
| **Text / Document** | `.txt`, `.pdf`, `.docx` | `roberta-base-openai-detector` token sequence classification with document extraction & lexical statistics |

---

## Key Features

- **Multi-Modal Verification**: Live neural and signal processing pipelines for image, video, audio, and text documents.
- **Cryptographic & Forensic Telemetry**: SHA-256 payload integrity hashing, execution runtime timing, resolution, duration, and lexical statistics.
- **Forensic Intelligence Dashboard (`/dashboard`)**:
  - Live summary cards: Total analyses, Authentic Media, AI Synthetic Media, Uncertain/Alerts.
  - Interactive Chart.js donut chart reflecting actual verdict distribution.
  - Modality breakdown visualizer (Image, Video, Audio, Document).
  - Recent verification activity table with direct view & PDF actions.
- **Persistent Verification History (`/history`)**:
  - SQLite database backing (`deepshield.db`) that persists across application restarts.
  - Keyword search across filename, SHA-256, and explanation text.
  - Multi-filtering by media type, verdict category, and verification date.
  - Sorting options (Newest, Oldest, Confidence High/Low, Filename A–Z).
  - Record deletion with native confirmation dialog.
- **Presentation-Ready PDF Certificates (`/result/<id>/pdf`)**:
  - Generates downloadable PDF reports with ReportLab containing artifact metadata, model architecture, confidence scores, forensic breakdown, and analytical disclaimers.

---

## Project Structure

```
DeepGuard-main/
├── app.py                     # Flask application & routing (Home, Dashboard, History, Result, PDF, Delete)
├── database.py                # SQLite persistence layer (schema, migrations, queries, telemetry)
├── deepshield.db              # Local SQLite database file (created automatically)
├── config.py                  # Upload configuration, allowed extensions, and file size limits
├── requirements.txt           # Python dependencies
├── models/
│   ├── image_detector.py      # CLIP zero-shot image detector
│   ├── video_detector.py      # EfficientNet + texture-artifact video analyzer
│   ├── audio_detector.py      # Acoustic heuristics + Wav2Vec2 audio analyzer
│   └── text_detector.py       # RoBERTa OpenAI text detector
├── utils/
│   ├── forensics.py           # SHA-256 checksums and authentic media metadata extraction
│   └── pdf_generator.py       # ReportLab PDF certificate generator
├── templates/
│   ├── index.html             # Verification portal & drag-and-drop upload
│   ├── dashboard.html         # Real-time intelligence dashboard & metrics
│   ├── history.html           # Audit trail with search, filters, sorting, and deletion
│   └── result.html            # Forensic diagnosis & PDF report download
└── static/
    ├── css/style.css
    ├── js/script.js
    └── uploads/               # Temporary analysis payload directory
```

---

## Database Schema (`verifications`)

| Column | Type | Description |
|---|---|---|
| `id` | `INTEGER PRIMARY KEY AUTOINCREMENT` | Unique verification audit ID |
| `filename` | `TEXT NOT NULL` | Uploaded artifact filename |
| `media_type` | `TEXT NOT NULL` | Modality: `image`, `video`, `audio`, `text` |
| `prediction` | `TEXT NOT NULL` | Detection verdict title |
| `verdict_category` | `TEXT NOT NULL` | Classification group: `ai`, `human`, `alert` |
| `ai_score` | `REAL NOT NULL` | AI synthetic probability percentage (0–100) |
| `human_score` | `REAL NOT NULL` | Human authenticity probability percentage (0–100) |
| `confidence` | `REAL NOT NULL` | Primary confidence metric |
| `reason` | `TEXT` | Forensic pipeline explanation |
| `file_size` | `INTEGER` | Payload file size in bytes |
| `status` | `TEXT NOT NULL` | Execution status: `completed`, `failed`, `warning` |
| `error_details` | `TEXT` | Diagnostic error information if applicable |
| `sha256` | `TEXT` | SHA-256 cryptographic hash of payload |
| `model_name` | `TEXT` | Specific model architecture and version |
| `processing_time` | `TEXT` | Actual inference execution duration |
| `metadata_json` | `TEXT` | Structured JSON containing forensic metadata |
| `created_at` | `TEXT` | Timestamp (`YYYY-MM-DD HH:MM:SS`) |

---

## Running Locally

```bash
# 1. Activate virtual environment
venv\Scripts\activate       # Windows PowerShell / CMD
# or on Linux/macOS: source venv/bin/activate

# 2. Install dependencies (if not already installed)
pip install -r requirements.txt

# 3. Launch application
python app.py
```

Access the application in your browser at `http://127.0.0.1:5000/`.

