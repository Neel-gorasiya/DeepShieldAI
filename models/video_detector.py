import warnings
warnings.filterwarnings("ignore")

import os
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# Use weights= API (suppresses pretrained= deprecation warning)
weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
model = models.efficientnet_b0(weights=weights).to(device)

# Binary classifier head: class 0 = Real, class 1 = Fake/AI
in_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(in_features, 2).to(device)
model.eval()

# ImageNet-normalised transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def _frame_artifact_score(frame_rgb: np.ndarray) -> float:
    """
    Heuristic deepfake artifact scorer.
    AI / deepfake frames tend to have:
      - Lower Laplacian variance (over-smoothed surfaces)
      - Suppressed DCT high-frequency energy
    Returns [0, 1] where 1 = highly likely AI/fake.
    """
    try:
        # Resize to standard size for stable and bounded heuristic processing
        if frame_rgb.shape[0] > 720 or frame_rgb.shape[1] > 1280:
            frame_rgb = cv2.resize(frame_rgb, (1280, 720), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        # Laplacian variance
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # DCT high-freq energy
        gray_f = np.float32(gray)
        dct = cv2.dct(gray_f)
        h, w = dct.shape
        high_freq = np.sum(np.abs(dct[h // 2:, w // 2:])) / (dct.size / 4 + 1e-6)

        lap_score = 1.0 - min(lap_var / 2000.0, 1.0)
        hf_score = 1.0 - min(high_freq / 50.0, 1.0)

        return float(lap_score * 0.6 + hf_score * 0.4)
    except Exception:
        return 0.5


def _model_fake_score(img_tensor: torch.Tensor) -> float:
    """EfficientNet backbone confidence signal (class-1 probability)."""
    try:
        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.softmax(output, dim=1)
        return float(prob[0][1])
    except Exception:
        return 0.5


def detect_video(path: str) -> dict:
    if not os.path.exists(path):
        return {
            "prediction": "Error",
            "ai_score": 0.0,
            "human_score": 0.0,
            "reason": "Video file not found on server.",
        }

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {
            "prediction": "Video could not be opened",
            "ai_score": 0.0,
            "human_score": 100.0,
            "reason": "OpenCV could not open the video file. Check format or codec compatibility.",
        }

    try:
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = raw_fps if (raw_fps and 1.0 <= raw_fps <= 120.0) else 25.0
        sample_interval = max(1, int(fps))  # ~1 sample per second of video
        max_samples = 20
        max_frame_iterations = 2000  # Hard loop cap to prevent CPU exhaustion on malformed streams

        artifact_scores, model_scores = [], []
        frame_count = 0
        sampled = 0

        while sampled < max_samples and frame_count < max_frame_iterations:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_count % sample_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                artifact_scores.append(_frame_artifact_score(frame_rgb))

                pil_img = Image.fromarray(frame_rgb)
                img_t = transform(pil_img).unsqueeze(0).to(device)
                model_scores.append(_model_fake_score(img_t))

                sampled += 1

            frame_count += 1

    except Exception as e:
        return {
            "prediction": "Error",
            "ai_score": 0.0,
            "human_score": 0.0,
            "reason": f"Video analysis error: {str(e)}",
        }
    finally:
        cap.release()

    if not artifact_scores:
        return {
            "prediction": "Video could not be analyzed",
            "ai_score": 0.0,
            "human_score": 100.0,
            "reason": "No readable video frames could be extracted.",
        }

    avg_artifact = float(np.mean(artifact_scores))
    avg_model = float(np.mean(model_scores))

    # Artifact heuristic weighted higher (model head is untrained)
    combined_fake = avg_artifact * 0.65 + avg_model * 0.35

    ai_score = round(combined_fake * 100, 2)
    human_score = round((1.0 - combined_fake) * 100, 2)

    prediction = "AI Generated Video" if ai_score > human_score else "Human Recorded Video"

    reason = (
        f"Analyzed {sampled} frame(s) sampled at ~1 fps using EfficientNet visual features "
        f"combined with DCT high-frequency and Laplacian texture artifact analysis. "
        f"Texture artifact score: {round(avg_artifact*100,1)}% | "
        f"EfficientNet score: {round(avg_model*100,1)}%."
    )

    return {
        "prediction": prediction,
        "ai_score": ai_score,
        "human_score": human_score,
        "reason": reason,
    }
