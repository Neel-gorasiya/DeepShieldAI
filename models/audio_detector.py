import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoFeatureExtractor, AutoModel
import torch
import librosa
import numpy as np

MODEL_NAME = "facebook/wav2vec2-base"

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


def _acoustic_artifact_score(audio: np.ndarray, sr: int) -> float:
    """
    Heuristic synthetic-speech scorer based on classic signal-processing cues.
    AI/TTS/voice-clone audio tends to have:
      - unnaturally low pitch (F0) variance / jitter
      - unnaturally smooth short-term energy (low shimmer)
      - suppressed high-frequency spectral detail (over-smooth vocoder output)
    Returns a value in [0, 1] where 1 = strongly resembles synthetic speech.
    """
    try:
        # Pitch (F0) variability
        f0, voiced_flag, _ = librosa.pyin(
            audio, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
        )
        f0_voiced = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        if len(f0_voiced) > 3:
            pitch_cv = np.std(f0_voiced) / (np.mean(f0_voiced) + 1e-6)
        else:
            pitch_cv = 0.15  # neutral fallback

        # Short-term energy variability (shimmer proxy)
        rms = librosa.feature.rms(y=audio)[0]
        shimmer = np.std(rms) / (np.mean(rms) + 1e-6) if len(rms) > 1 else 0.2

        # High-frequency spectral detail
        spec = np.abs(librosa.stft(audio))
        freq_bins = spec.shape[0]
        high_freq_energy = np.mean(spec[freq_bins // 2:, :])
        total_energy = np.mean(spec) + 1e-6
        hf_ratio = high_freq_energy / total_energy

        pitch_score = 1.0 - min(pitch_cv / 0.35, 1.0)     # low pitch variance -> more synthetic
        shimmer_score = 1.0 - min(shimmer / 0.6, 1.0)     # low energy variance -> more synthetic
        hf_score = 1.0 - min(hf_ratio / 0.35, 1.0)        # low HF detail -> more synthetic

        return float(pitch_score * 0.4 + shimmer_score * 0.3 + hf_score * 0.3)
    except Exception:
        return 0.5


def _embedding_variance_score(audio: np.ndarray, sr: int) -> float:
    """
    Secondary signal from wav2vec2 hidden-state variance. Real recordings tend to
    carry more frame-to-frame representational variability (background noise, room
    tone, breath) than clean synthetic speech. Not a trained classifier - a rough
    supporting signal only, kept as a minority weight in the blend.
    """
    try:
        # Limit audio snippet for embedding inference to 10 seconds to conserve memory
        sample_audio = audio[: sr * 10]
        inputs = feature_extractor(sample_audio, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state  # (1, T, D)

        frame_std = hidden.squeeze(0).std(dim=0).mean().item()
        # Normalise into [0,1]; low variance -> more "synthetic-looking" embeddings
        return float(1.0 - min(frame_std / 1.5, 1.0))
    except Exception:
        return 0.5


def detect_audio(path: str) -> dict:
    if not os.path.exists(path):
        return {
            "prediction": "Error",
            "ai_score": 0.0,
            "human_score": 0.0,
            "reason": "Audio file not found on server.",
        }

    try:
        # Bound audio loading to max 60 seconds to prevent RAM exhaustion
        audio, sr = librosa.load(path, sr=16000, duration=60.0)
        audio = np.array(audio)

        if len(audio) < sr * 0.5:
            return {
                "prediction": "Audio too short to analyze",
                "ai_score": 0.0,
                "human_score": 100.0,
                "reason": "Provide at least half a second of audio for reliable analysis.",
            }

        artifact_score = _acoustic_artifact_score(audio, sr)
        embed_score = _embedding_variance_score(audio, sr)

        # Artifact heuristics weighted higher; embedding variance is a supporting signal
        combined_fake = artifact_score * 0.7 + embed_score * 0.3

        ai_score = round(combined_fake * 100, 2)
        human_score = round((1.0 - combined_fake) * 100, 2)

        prediction = "AI Generated Voice" if ai_score > human_score else "Human Voice"

        reason = (
            f"Analyzed pitch stability, energy (shimmer) variance, high-frequency spectral "
            f"detail, and wav2vec2 embedding variability. "
            f"Acoustic artifact score: {round(artifact_score * 100, 1)}% | "
            f"Embedding variance score: {round(embed_score * 100, 1)}%."
        )

        return {
            "prediction": prediction,
            "ai_score": ai_score,
            "human_score": human_score,
            "reason": reason,
        }

    except Exception as e:
        return {
            "prediction": "Error",
            "ai_score": 0.0,
            "human_score": 0.0,
            "reason": f"Audio processing failed: {str(e)}",
        }
