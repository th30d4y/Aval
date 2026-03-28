# அவள் — Aval 🛡️

<p align="center">
  <img src="https://img.shields.io/badge/Python-Backend-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-ML%20Models-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/HTML-Frontend-E34F26?style=for-the-badge&logo=html5&logoColor=white" />
  <img src="https://img.shields.io/badge/Purpose-Women%20Safety-ff69b4?style=for-the-badge" />
</p>

> **அவள்** *(Tamil: "She")* — An AI-powered women safety system combining machine learning threat detection with real-time emergency response, built to protect and empower.

---

## 📁 Project Structure

```
Aval/
├── backend/
│   ├── Training/
│   │   ├── Training.ipynb               # Main model training notebook
│   │   └── darktraining.ipynb           # Low-light / night scenario training
│   ├── models/
│   │   ├── women_safety_complete_model  # Full trained model (Keras/H5)
│   │   ├── women_safety_complete_model  # Alternate export format
│   │   ├── women_safety_mobile_integrati# TFLite model for mobile inference
│   │   └── women_safety_model_config.json
│   └── app.py                           # Python backend server (Flask/FastAPI)
├── frontend/
│   └── index.html                       # Web frontend
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ✨ Features

### 🆘 SOS & Emergency Alert
- One-tap SOS to instantly notify emergency contacts
- Sends real-time GPS coordinates along with the alert
- Triggers automatically on shake detection

### 📍 Live Location Sharing
- Continuous real-time location updates to trusted contacts
- Background location tracking during active SOS

### 📳 Shake to Trigger Alert
- Detects sudden shake gestures via accelerometer
- Activates SOS without requiring the phone to be unlocked

### 🤖 ML-Based Threat Detection
- Trained on women safety scenarios including low-light and night conditions (`darktraining.ipynb`)
- Complete model available in full and mobile-optimised (TFLite) formats
- Configurable via `women_safety_model_config.json`

### 👥 Crowd Analysis
- Detects unsafe crowd patterns and density using computer vision
- Alerts user when entering a potentially dangerous zone

### 🏥 Nearby Police / Hospital Finder
- Fetches nearby police stations and hospitals using GPS
- Provides distance, contact info, and navigation

### 🛠️ Admin Panel
- View SOS alert logs and incident history
- Monitor active sessions and user data (with consent)

### 🥋 Self-Defense Tips & Videos
- Curated self-defense guides accessible from the frontend

---

## 🧠 ML Models

| File | Description |
|---|---|
| `women_safety_complete_model` | Full trained model (Keras / H5 format) |
| `women_safety_complete_model` (alt) | Secondary export format for serving |
| `women_safety_mobile_integrati...` | TFLite — optimised for mobile inference |
| `women_safety_model_config.json` | Model configuration and class mappings |

### Training Notebooks

| Notebook | Purpose |
|---|---|
| `Training.ipynb` | Main model training pipeline |
| `darktraining.ipynb` | Training on low-light / night-time scenarios |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python (Flask / FastAPI) |
| ML Framework | TensorFlow / Keras + TFLite |
| Training | Jupyter Notebook |
| Frontend | HTML5 |
| Location Services | Google Maps API / Browser Geolocation |
| Notifications | SMS / Firebase Cloud Messaging |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip
- Jupyter Notebook (for training)
- A modern web browser (for frontend)

### Installation

#### 1. Clone the repository
```bash
git clone https://github.com/th30d4y/Aval.git
cd Aval
```

#### 2. Install dependencies
```bash
pip install -r requirements.txt
```

#### 3. Run the backend server
```bash
cd backend
python app.py
```

#### 4. Open the frontend
Open `frontend/index.html` in your browser, or serve it:
```bash
cd frontend
python -m http.server 8080
```

#### 5. (Optional) Retrain the model
```bash
cd backend/Training
jupyter notebook Training.ipynb
# For low-light training:
jupyter notebook darktraining.ipynb
```

---

## 📦 Dependencies

Install all required packages with:

```bash
pip install -r requirements.txt
```

Key dependencies include TensorFlow, Flask/FastAPI, OpenCV, and NumPy. Refer to `requirements.txt` for the full list.

---

## 👥 Contributors

- **Stalin-143** — Stalin
- **harriiinnii**

---

## 📄 License

This project is open source. See the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <strong>அவள் பாதுகாப்பாக இருக்கட்டும்</strong><br/>
  <em>May She Be Safe</em>
</p>
