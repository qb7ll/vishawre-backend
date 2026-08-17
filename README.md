# VishAware — Backend

Flask API and AI voice-detection service for the VishAware platform — an AI voice phishing (vishing) detection and training system.

Senior graduation project — Information Technology Department, King Abdulaziz University (2025–2026).

**Team:** Fatima Tariq Hajjar, Mayar Sameer Aljudaybi, Razan Sameer Alshaikh
**Supervisor:** Dr. Saja Alqurashi
**Frontend repo:** [vishawre](https://github.com/qb7ll/vishawre)

## What it does

- REST API for authentication, session management, and voice-sample handling
- Serves the AI voice-detection model that classifies audio as real or AI-generated
- Stores user accounts, training sessions, and results in a SQLite database

## Tech stack

- **Flask** (Python) — API server
- **SQLite** — data storage
- **PyTorch**, **Hugging Face Transformers**, **Librosa** — audio feature extraction and voice classification model
- Token-based authentication with SHA-256 password hashing

## Model performance

The voice-classification model achieves **97% detection accuracy** distinguishing real human voices from AI-generated ones. See `models/deepfak-audio_detection_final/evaluation_report.json` for full metrics.

## Project structure

```
app.py                              Main Flask application & API routes
deepfake_detector.py                Audio preprocessing & model inference
models/deepfak-audio_detection_final/
  config.json                       Model configuration
  evaluation_report.json            Accuracy / precision / recall metrics
  confusion_matrix.png              Evaluation confusion matrix
requirements.txt                    Python dependencies
```

## Note on data files

The SQLite database (`database.db`), uploaded audio samples (`uploads/`), and test recordings are excluded from this repository — they contained test/demo user data and are not needed to run or review the code. A fresh database is created automatically on first run.

## Getting started

```bash
pip install -r requirements.txt
python app.py
```

---

Part of a senior project exploring the intersection of AI security, network security, and full-stack development.
