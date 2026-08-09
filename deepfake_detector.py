"""
Deepfake voice detection service.

This module loads a lightweight local Hugging Face audio classification model
and exposes a small API that the Flask backend can use for inference.
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass

import librosa
import torch
from huggingface_hub import snapshot_download
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AVAILABLE_MODELS = {
    "Vansh180/deepfake-audio-wav2vec2": {
        "type": "huggingface",
        "path": os.path.join(BASE_DIR, "backend", "models", "deepfake-audio-wav2vec2")
    },
    "MelodyMachine/Deepfake-audio-detection-V2": {
        "type": "huggingface",
        "path": os.path.join(BASE_DIR, "backend", "models", "Deepfake-audio-detection-V2")
    },
    "local/deepfak-audio_detection_final": {
        "type": "local",
        "path": os.path.join(BASE_DIR, "backend", "models", "deepfak-audio_detection_final")
    }
}
DEFAULT_MODEL = "Vansh180/deepfake-audio-wav2vec2"

WINDOW_SECONDS = 3.0
WINDOW_OVERLAP_SECONDS = 1.0
MIN_AUDIO_SECONDS = 1.5

@dataclass
class DetectorMetadata:
    model_id: str
    model_dir: str
    downloaded: bool
    ready: bool
    device: str
    message: str


class DeepfakeVoiceDetector:
    def __init__(self):
        self._runtime_loaded = False
        self._models = {}
        self._feature_extractors = {}
        self._torch = torch
        self._librosa = librosa
        self._snapshot_download = snapshot_download
        self._device = "cpu"
        self._lock = threading.Lock()

    def _load_runtime(self):
        if self._runtime_loaded:
            return

        self._AutoFeatureExtractor = AutoFeatureExtractor
        self._AutoModelForAudioClassification = AutoModelForAudioClassification
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._runtime_loaded = True

    def ensure_model_downloaded(self, model_id: str):
        if model_id not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model_id: {model_id}")
            
        self._load_runtime()
        config = AVAILABLE_MODELS[model_id]
        model_dir = config["path"]
        os.makedirs(model_dir, exist_ok=True)

        if os.path.exists(os.path.join(model_dir, "config.json")):
            return model_dir

        if config["type"] == "local":
            raise ValueError(f"Local model missing config.json in {model_dir}")

        return self._snapshot_download(repo_id=model_id, local_dir=model_dir, local_dir_use_symlinks=False, ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.bin"])

    def load(self, model_id: str = None):
        if not model_id:
            model_id = DEFAULT_MODEL
            
        with self._lock:
            if model_id in self._models:
                return

            self._load_runtime()
            model_dir = self.ensure_model_downloaded(model_id)

            extractor = self._AutoFeatureExtractor.from_pretrained(model_dir)
            model = self._AutoModelForAudioClassification.from_pretrained(
                model_dir,
                low_cpu_mem_usage=True,
            )
            model.to(self._device)
            model.eval()
            
            self._feature_extractors[model_id] = extractor
            self._models[model_id] = model

    def get_metadata(self, model_id: str = None) -> DetectorMetadata:
        if not model_id:
            model_id = DEFAULT_MODEL
            
        config = AVAILABLE_MODELS.get(model_id)
        if not config:
            return DetectorMetadata(model_id, "", False, False, "cpu", "unknown_model")
            
        model_dir = config["path"]
        downloaded = os.path.exists(os.path.join(model_dir, "config.json"))
        ready = downloaded and model_id in self._models
        message = "model_ready" if ready else "model_downloaded" if downloaded else "model_missing"
        
        return DetectorMetadata(
            model_id=model_id,
            model_dir=model_dir,
            downloaded=downloaded,
            ready=ready,
            device=self._device,
            message=message,
        )

    def _normalize_label(self, label: str, index: int) -> str:
        cleaned = (label or "").strip().lower()
        if any(token in cleaned for token in ("real", "bonafide", "genuine", "human")):
            return "real"
        if any(token in cleaned for token in ("fake", "spoof", "synthetic", "ai")):
            return "fake"
        return "real" if index == 0 else "fake"

    def _build_windows(self, audio):
        sample_rate = 16000
        window_size = int(WINDOW_SECONDS * sample_rate)
        step_size = int(max(1, (WINDOW_SECONDS - WINDOW_OVERLAP_SECONDS) * sample_rate))

        if len(audio) <= window_size:
            return [audio]

        windows = []
        for start in range(0, len(audio), step_size):
            end = start + window_size
            chunk = audio[start:end]
            if len(chunk) < sample_rate:
                break
            windows.append(chunk)
            if end >= len(audio):
                break
        return windows or [audio[:window_size]]

    def analyze(self, file_path: str, model_id: str = None):
        if not model_id:
            model_id = DEFAULT_MODEL
            
        if model_id not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model_id: {model_id}")
            
        self.load(model_id)
        model = self._models[model_id]
        extractor = self._feature_extractors[model_id]

        audio, sample_rate = self._librosa.load(file_path, sr=16000, mono=True)
        duration_seconds = len(audio) / sample_rate if sample_rate else 0

        if duration_seconds < MIN_AUDIO_SECONDS:
            raise ValueError("Audio clip is too short. Please upload at least 1.5 seconds.")

        windows = self._build_windows(audio)
        aggregated_probs = []

        with self._torch.no_grad():
            for window in windows:
                inputs = extractor(
                    window,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {key: value.to(self._device) for key, value in inputs.items()}
                outputs = model(**inputs)
                probs = self._torch.softmax(outputs.logits, dim=-1)[0]
                aggregated_probs.append(probs.detach().cpu())

        stacked = self._torch.stack(aggregated_probs)
        mean_probs = stacked.mean(dim=0)

        id2label = getattr(model.config, "id2label", {}) or {}
        label_map = {
            self._normalize_label(label, index): float(mean_probs[index].item())
            for index, label in id2label.items()
        }

        real_score = label_map.get("real")
        fake_score = label_map.get("fake")

        if real_score is None or fake_score is None:
            real_score = float(mean_probs[0].item())
            fake_score = float(mean_probs[1].item()) if mean_probs.shape[0] > 1 else 1.0 - real_score

        prediction = "fake" if fake_score >= real_score else "real"
        confidence = max(real_score, fake_score)

        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "probabilities": {
                "real": round(real_score, 4),
                "fake": round(fake_score, 4),
            },
            "segmentsAnalyzed": len(windows),
            "durationSeconds": round(duration_seconds, 2),
            "model": {
                "id": model_id,
                "localPath": AVAILABLE_MODELS[model_id]["path"],
                "device": self._device,
            },
            "thresholds": {
                "strong": 0.8,
                "moderate": 0.65,
            },
            "riskLevel": self._risk_level(confidence, prediction),
        }

    def _risk_level(self, confidence: float, prediction: str) -> str:
        # If prediction is real, the file is NOT high risk.
        if prediction == "real":
            return "low"
            
        if confidence >= 0.8:
            return "high"
        if confidence >= 0.65:
            return "medium"
        return "low"

detector_service = DeepfakeVoiceDetector()
