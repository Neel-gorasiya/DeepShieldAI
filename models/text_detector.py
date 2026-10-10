import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

from transformers import pipeline
import pdfplumber
import docx

detector = pipeline(
    "text-classification",
    model="roberta-base-openai-detector",
    truncation=True,
    max_length=512,
)

MAX_EXTRACT_CHARS = 50_000


def extract_text(file_path: str) -> str:
    if not os.path.exists(file_path):
        return ""

    file_ext = os.path.splitext(file_path)[1].lower()
    text = ""

    try:
        if file_ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read(MAX_EXTRACT_CHARS)

        elif file_ext == ".pdf":
            pages_text = []
            with pdfplumber.open(file_path) as pdf:
                # Limit to first 50 pages to prevent denial of service
                for idx, page in enumerate(pdf.pages[:50]):
                    page_content = page.extract_text() or ""
                    pages_text.append(page_content)
                    if sum(len(p) for p in pages_text) >= MAX_EXTRACT_CHARS:
                        break
            text = "\n".join(pages_text)[:MAX_EXTRACT_CHARS]

        elif file_ext == ".docx":
            doc = docx.Document(file_path)
            # Limit to first 500 paragraphs
            paragraphs = [para.text for para in doc.paragraphs[:500] if para.text]
            text = "\n".join(paragraphs)[:MAX_EXTRACT_CHARS]

    except Exception:
        text = ""

    return text.strip()


def detect_text(path: str) -> dict:
    if not os.path.exists(path):
        return {
            "prediction": "Error",
            "ai_score": 0.0,
            "human_score": 0.0,
            "reason": "Document file not found on server.",
        }

    try:
        text = extract_text(path)

        if not text:
            return {
                "prediction": "Unable to read document",
                "ai_score": 0.0,
                "human_score": 0.0,
                "reason": "Document text could not be extracted or file was empty.",
            }

        result = detector(text[:2000])[0]

        # roberta-base-openai-detector label map: 0 = "Fake" (AI-generated), 1 = "Real" (human-written)
        if result["label"] == "Fake":
            ai_score = result["score"] * 100.0
            human_score = 100.0 - ai_score
            prediction = "AI Generated Text"
        else:
            human_score = result["score"] * 100.0
            ai_score = 100.0 - human_score
            prediction = "Human Written Text"

        return {
            "prediction": prediction,
            "ai_score": round(ai_score, 2),
            "human_score": round(human_score, 2),
            "reason": "RoBERTa language model detected AI writing patterns.",
        }

    except Exception as e:
        return {
            "prediction": "Error",
            "ai_score": 0.0,
            "human_score": 0.0,
            "reason": f"Text analysis failed: {str(e)}",
        }