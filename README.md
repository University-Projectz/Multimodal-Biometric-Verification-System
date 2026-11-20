# Multimodal Biometric Verification System (Educational Project)

## Overview
This project is an **educational, end-to-end multimodal biometric verification system**.  
It demonstrates practical machine learning workflows by combining multiple biometric and audio modalities:

- Face Recognition
- Voice Recognition
- Fingerprint Recognition
- Command Recognition (GMM/HMM)
- Early Fusion (feature-level multimodal fusion)

The project focuses on both:
1. **Modeling and evaluation**, and  
2. **Educational clarity** through structured notebooks and runnable applications.

---

## Project Objectives
- Build complete pipelines from data preparation to model evaluation.
- Compare different methods (e.g., **GMM vs HMM** for command recognition).
- Integrate trained components into runnable apps.
- Keep the repository clean and safe for GitHub publishing.

---

## Educational Scope
This repository is designed for learning and demonstration:

- Step-by-step notebooks by modality.
- Clear explanations of what is done in each stage.
- Structured flow from raw data to deployment-oriented inference apps.
- Notebook sections standardized for readability and teaching.

---

## Implemented Pipelines

### 1) Face Pipeline
- Dataset inspection
- Preprocessing
- Face detection
- Feature extraction
- Machine learning
- YOLO experiment

### 2) Voice Pipeline
- Audio preprocessing/standardization
- Voice feature extraction
- Voice model training/evaluation

### 3) Fingerprint Pipeline
- Dataset inspection
- Preprocessing/enhancement
- Feature extraction
- Model training/evaluation

### 4) Command Recognition Pipeline
- Feature extraction from command audio
- GMM/HMM training
- Performance comparison (accuracy, WER, confusion matrix)
- Visual diagnostics

### 5) Early Fusion Pipeline
- Multi-modal feature loading
- Feature-level concatenation
- Fusion model training/evaluation
- Streamlit inference interface

---

## Project Structure

```text
Multimodal Biometric Verification System/
│
├── src/
│   ├── apps/
│   │   ├── live_camera_app.py
│   │   └── fusion_streamlit_app.py
│   │
│   ├── Face/
│   │   ├── face_camera.py
│   │   ├── face_feature_extraction.py
│   │   ├── face_processing.py
│   │   ├── face_verification.py
│   │   └── webcam_test.py
│   │
│   ├── Voice/
│   │   ├── voice_recognition.py
│   │   ├── hierarchical_verification.py
│   │   └── feature_fusion.py
│   │
│   ├── Command_Recognition/
│   │   ├── command_recognition.py
│   │   └── record_commands.py
│   │
│   └── Early_Fusion/
│       ├── extractors.py
│       └── train_fusion.py
│
├── Notebooks/
│   ├── Face/
│   ├── Voice/
│   ├── Fingerprint/
│   ├── Command_Recognition/
│   └── Early_Fusion/
│
├── Dataset/          # local/private (ignored in GitHub)
│   ├── Raw/
│   ├── Processed/
│   └── YOLO_Experiment/
│
├── Models/           # local/private (ignored in GitHub)
│   ├── Face/
│   ├── Voice/
│   ├── Fingerprint/
│   ├── Command_Recognition/
│   └── Early_Fusion/
│
├── run_live_camera.bat
├── run_streamlit_fusion.bat
├── requirements.txt
└── .gitignore
```

---

## Installation

## 1) Create environment (recommended)
```bash
python -m venv .venv
```

### Windows
```bash
.venv\Scripts\activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

---

## Run Applications

### 1) Live Camera App
```bash
run_live_camera.bat
```
or
```bash
python src/apps/live_camera_app.py
```

### 2) Streamlit Fusion App
```bash
run_streamlit_fusion.bat
```
or
```bash
streamlit run src/apps/fusion_streamlit_app.py
```

---

## Suggested Learning Path
1. Face notebooks (01 → end)
2. Voice notebooks (01 → end)
3. Fingerprint notebooks (01 → end)
4. Command Recognition notebook
5. Early Fusion notebook
6. App integration (`src/apps/`)

---

## Disclaimer
This is an **educational project** for learning, experimentation, and workflow demonstration.  
It is not a production-ready biometric security product.
