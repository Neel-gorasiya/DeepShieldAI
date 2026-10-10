# Project Understanding — DeepShield AI (AI & Synthetic Media Detection System)

This document is my own explanation of how DeepShield AI works, why it's built the way
it is, and what I learned putting it together. I'm keeping this in the repo mainly
for two reasons: so anyone reviewing the project (a professor, a recruiter, a
teammate) can see the reasoning behind the design, and so I have something honest
to say in a viva or interview instead of just describing the code from memory.

## 1. The problem this project is solving

AI-generated media — fake images, cloned voices, synthetic video, and
LLM-written text — has become easy to produce and hard to tell apart from the
real thing by eye. DeepShield AI's goal is to give a quick, explainable first-pass
signal: upload a file, get back a prediction ("AI Generated" vs "Human
Created/Written"), an AI-likelihood score, a human-likelihood score, and a short
reason describing *what* was analyzed. It's a single Flask app that routes any of
four media types to a dedicated detector module.

## 2. High-level architecture

```
Browser (index.html)
   │  uploads a file
   ▼
Flask app.py  →  routes by file extension (config.py holds the allowed lists)
   │
   ├── .jpg/.png/.jpeg   → models/image_detector.py
   ├── .mp4/.avi/.mov    → models/video_detector.py
   ├── .wav/.mp3         → models/audio_detector.py
   └── .txt/.pdf/.docx   → models/text_detector.py
   │
   ▼
result.html  →  shows prediction + AI score / human score bars + reason text
```

Each detector module is self-contained: it loads its own model(s) once at import
time (so the model isn't reloaded on every request), exposes a single
`detect_<type>(path)` function, and always returns the same shape of dictionary:
`{"prediction": ..., "ai_score": ..., "human_score": ..., "reason": ...}`. That
consistent contract is what lets `app.py` stay generic — it doesn't need to know
anything about *how* each detector works, only that it returns this dictionary.

## 3. How each detector actually works

### Image — CLIP zero-shot classification
CLIP (Contrastive Language–Image Pretraining) was trained to place images and text
into a shared embedding space, so it can score how well an image matches a text
prompt without ever being trained on a "real vs AI" label. The image detector
encodes the uploaded image and two text prompts — `"real photo"` and `"ai
generated image"` — into that shared space, then measures cosine similarity
between the image and each prompt. Softmax over those two similarities gives a
human-score and an ai-score that sum to 100. This is a *zero-shot* trick, not a
purpose-trained classifier, which is why it's a reasonable heuristic rather than
a highly accurate one — CLIP has no explicit concept of "deepfake artifacts," it's
just pattern-matching against the vibe of those two phrases.

### Video — EfficientNet features + texture/frequency artifacts
Video is sampled at roughly 1 frame per second (up to 20 frames) rather than
processed frame-by-frame, which keeps inference fast. Each sampled frame is
scored two ways and the two scores are blended:
- **EfficientNet-B0** (ImageNet-pretrained) with its final layer swapped for a
  fresh 2-class linear head. Because that head was never trained on real/fake
  video data, its output is a weak signal, so it only gets 35% weight.
- **A hand-written artifact heuristic** using Laplacian variance (blurred /
  over-smoothed surfaces score lower) and DCT high-frequency energy (AI-generated
  frames tend to suppress fine high-frequency detail during generation/upscaling).
  This needs no training data at all and gets 65% of the weight, since it's the
  more trustworthy signal here.

### Audio — acoustic artifacts + Wav2Vec2 embedding variance
This module went through the most rework (see the bug-fix section below). The
final version doesn't rely on a trained classifier at all — instead it computes
three classic acoustic cues that are known proxies for synthetic/TTS speech:
- **Pitch (F0) stability** — real human speech has natural pitch wobble;
  synthetic speech tends to be unnaturally steady.
- **Energy/shimmer variance** — the same idea applied to loudness over time.
- **High-frequency spectral detail** — vocoders used in speech synthesis tend to
  under-represent very high frequencies compared to a real microphone recording.

These three are combined into an "artifact score," which is blended with a
secondary signal from **Wav2Vec2** — not used as a classifier, but purely for the
*frame-to-frame variance of its hidden-state embeddings*, on the idea that real
recordings carry more incidental variability (room tone, breath, background
noise) than clean synthetic audio.

### Text — RoBERTa fine-tuned as an "OpenAI text detector"
`roberta-base-openai-detector` is a RoBERTa model OpenAI released alongside GPT-2,
fine-tuned specifically to tell GPT-2 output apart from human text. Its output
label is `"Fake"` for AI-generated and `"Real"` for human-written, with a
confidence score attached to whichever label it predicted. Text is extracted from
`.txt`, `.pdf` (via `pdfplumber`), or `.docx` (via `python-docx`) first, then fed
through the model (with proper tokenizer-level truncation to 512 tokens).

## 4. Bugs I found and fixed while preparing this for GitHub

Going through the code carefully before publishing turned up a few real
functional problems — worth calling out explicitly, since understanding *why*
something was broken is more valuable than just knowing it now works:

1. **Text detector's prediction was backwards.** The code checked
   `if result["label"] == "Fake": prediction = "Human Written Text"` — but
   `"Fake"` from this specific model means AI-generated, not human. So every
   single text prediction was inverted. On top of that, the confidence score was
   always written into `ai_score` no matter which label won, so even the
   percentages were wrong on the "Real" side. This was the most serious bug,
   because it silently gave the *opposite* answer with high confidence — worse
   than giving no answer at all.

2. **Audio detector's model signal was pure noise.** `facebook/wav2vec2-base` is
   a base speech-representation model — it has no fine-tuned head for real/fake
   classification. Loading it through `AutoModelForAudioClassification` (as the
   original code did) attaches a randomly initialized linear layer on top, so
   "confidence" numbers coming out of it don't correspond to anything meaningful,
   and which output index means "real" vs "fake" is undefined for an untrained
   head. I replaced that approach with the acoustic-heuristic + embedding-variance
   blend described above, which doesn't depend on a trained head at all.

3. **`openai-clip` failed to import on a fresh install.** On Python 3.11/3.12,
   `import clip` pulls in `pkg_resources`, which modern `setuptools` no longer
   ships by default — so a clean `pip install -r requirements.txt` followed by
   running the app crashed immediately on the image detector. Fixed by pinning
   `setuptools<81` in `requirements.txt`.

4. **`config.py` existed but was never actually used.** `app.py` hardcoded its
   own copies of the allowed-extension lists and upload folder path instead of
   importing them from `config.py`, and `MAX_FILE_SIZE` was defined but never
   applied anywhere — so there was effectively no upload size limit. Now `app.py`
   imports everything from `config.py` and sets Flask's `MAX_CONTENT_LENGTH`.

5. **Missing-file and unsafe-filename handling.** `request.files["file"]` would
   throw an unhandled `KeyError` (crashing with a 500 error) if the form was
   submitted with no file attached, and the raw uploaded filename was used
   directly to build a filesystem path — a path-traversal risk if someone
   uploaded a file named something like `../../etc/passwd.jpg`. Fixed with
   `request.files.get(...)` plus a friendly fallback, and `secure_filename()`
   from Werkzeug.

## 5. What I'd call out honestly in a viva/interview

- This is a **multi-signal heuristic system**, not a state-of-the-art forensic
  detector. Two of the four detectors (image, and to a lesser extent text) lean
  on models trained for a related-but-different task (CLIP's general
  image-text matching, GPT-2-era text detection) rather than something
  purpose-trained on today's deepfake generators.
- The video and audio detectors intentionally combine a weak/absent learned
  signal with classical signal-processing heuristics, because there was no
  labeled real/fake dataset available to actually train those heads — a common
  and honest workaround at student-project scale, but a design decision I should
  be able to explain rather than hide.
- **Fixing the text-detector label inversion was the single highest-impact
  fix** — an inverted binary classifier is *worse than random*, because it's
  confidently wrong rather than uncertain.
- Given more time, the natural next steps would be: fine-tuning the video/audio
  heads on an actual labeled deepfake dataset (e.g. FaceForensics++ for video,
  ASVspoof for audio), adding face-level analysis for video instead of
  whole-frame analysis, and swapping the text detector for a model that also
  covers modern LLM output (GPT-2-era detectors are noticeably weaker against
  GPT-4/Claude/Gemini-generated text).

## 6. Setup & running (see README.md for full details)

```bash
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000/`, upload an image/video/audio/text file, and view the
result page with the prediction, score breakdown, and reasoning.
