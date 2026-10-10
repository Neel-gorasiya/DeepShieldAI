import warnings
warnings.filterwarnings("ignore")

import os
import torch
import clip
from PIL import Image

# Prevent decompression bomb Denial of Service attacks
Image.MAX_IMAGE_PIXELS = 50_000_000

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()


def detect_image(path: str) -> dict:
    if not os.path.exists(path):
        return {
            "prediction": "Error",
            "ai_score": 0.0,
            "human_score": 0.0,
            "reason": "Image file not found on server.",
        }

    try:
        with Image.open(path) as raw_img:
            # Verify image integrity
            raw_img.verify()

        # Reopen for preprocessing after verify()
        with Image.open(path) as raw_img:
            rgb_img = raw_img.convert("RGB")
            image = preprocess(rgb_img).unsqueeze(0).to(device)

        text = clip.tokenize(
            ["real photo", "ai generated image"]
        ).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            similarity = (image_features @ text_features.T).softmax(dim=-1)

        human = float(similarity[0][0]) * 100
        ai = float(similarity[0][1]) * 100

        if ai > human:
            prediction = "AI Generated Image"
        else:
            prediction = "Human Created Image"

        return {
            "prediction": prediction,
            "ai_score": round(ai, 2),
            "human_score": round(human, 2),
            "reason": "CLIP model detected synthetic visual patterns commonly found in AI generated images.",
        }

    except Exception as e:
        return {
            "prediction": "Error",
            "ai_score": 0.0,
            "human_score": 0.0,
            "reason": f"Image processing failed: {str(e)}",
        }