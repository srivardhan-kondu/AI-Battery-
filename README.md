# 🔋 AI-Based Battery Recycling System

> An intelligent, full-stack web application that automates battery identification, classification, and recyclable material analysis using a **3-stage computer vision pipeline**, a fine-tuned deep learning model, and a comprehensive material recovery database.

---

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)](https://pytorch.org/)
[![MobileNetV2](https://img.shields.io/badge/Model-MobileNetV2-orange)](https://arxiv.org/abs/1801.04381)
[![Accuracy](https://img.shields.io/badge/Val%20Accuracy-99.67%25-brightgreen)](/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

</div>

---

## 📋 Abstract

This project addresses the growing challenges in battery recycling. The current manual sorting process is inefficient, labour-intensive, and poses significant safety risks, hindering effective material recovery.

We introduce a new framework designed to **automate and de-risk** this crucial step. Our method uses an advanced two-stage computer vision pipeline to rapidly identify and categorize a diverse range of batteries from a single mixed stream. By linking these classifications to a detailed material database, the system provides a **real-time summary of recoverable resources**.

This approach offers a scalable, efficient, and safer alternative to traditional methods, ultimately improving the circular economy for battery recycling and enhancing resource recovery.

---

## 🛠️ Tech Stack

### Backend

| Technology | Version | Role |
|-----------|---------|------|
| **Python** | 3.9+ | Core programming language |
| **Flask** | 3.0.2 | Lightweight WSGI web framework for REST APIs and page routing |
| **Flask-SQLAlchemy** | 3.1.1 | ORM for database models (User accounts, session data) |
| **Flask-JWT-Extended** | 4.6.0 | JSON Web Token authentication with token blocklist |
| **Flask-Bcrypt** | 1.0.1 | Secure password hashing (bcrypt, cost factor 12) |
| **Flask-CORS** | 4.0.0 | Cross-Origin Resource Sharing for API access |
| **SQLite** | Built-in | Serverless relational database (zero configuration) |
| **Werkzeug** | 3.0.1 | WSGI utility library (secure file handling, routing) |

### Machine Learning & Computer Vision

| Technology | Version | Role |
|-----------|---------|------|
| **PyTorch** | 2.2+ | Deep learning framework for model training and inference |
| **TorchVision** | 0.17+ | Pre-trained MobileNetV2, image transforms, data loaders |
| **MobileNetV2** | — | CNN backbone fine-tuned for battery detection (99.67% accuracy) |
| **OpenCV** | 4.9+ | Computer vision: Hough transforms, Canny edge detection, contour analysis |
| **Tesseract OCR** | 5.x | Optical character recognition engine for text extraction |
| **pytesseract** | 0.3.10 | Python wrapper for Tesseract OCR |
| **Pillow** | 10.2+ | Image pre-processing: resize, sharpen, contrast enhancement |
| **Ultralytics** | 8.1+ | YOLO utilities and model management |

### Data Science & Analysis

| Technology | Version | Role |
|-----------|---------|------|
| **NumPy** | 1.26+ | Numerical operations, array processing |
| **Pandas** | 2.2+ | Data manipulation and analysis |
| **scikit-learn** | 1.4+ | ML utilities, metrics, data splitting |
| **Matplotlib** | 3.8+ | Training visualization, loss/accuracy plots |
| **Seaborn** | 0.13+ | Statistical data visualization |

### Frontend

| Technology | Role |
|-----------|------|
| **HTML5** | Page structure and semantic markup |
| **CSS3** | Professional light theme with responsive design |
| **Vanilla JavaScript** | API calls (fetch), DOM manipulation, drag-and-drop upload |
| **Inter Font** | Clean, professional typography (Google Fonts) |
| **JetBrains Mono** | Monospaced font for data values and code |

### Infrastructure & Tools

| Technology | Role |
|-----------|------|
| **SQLite** | Zero-config database — no external DB server needed |
| **JWT (JSON Web Tokens)** | Stateless authentication with 24-hour expiry |
| **bcrypt** | Industry-standard password hashing |
| **REST API** | JSON-based API architecture for all endpoints |
| **Git / GitHub** | Version control and code hosting |
| **venv** | Python virtual environment for dependency isolation |

### Why These Choices?

| Decision | Rationale |
|----------|-----------|
| **Flask over Django** | Lightweight, minimal boilerplate — ideal for API-first ML apps |
| **MobileNetV2 over ResNet/VGG** | 10× smaller model (8.7 MB), fast CPU inference (~200ms), high accuracy |
| **SQLite over PostgreSQL** | Zero setup, self-contained — perfect for single-server deployment |
| **Tesseract over cloud OCR** | Free, offline, no API keys required, runs entirely on-device |
| **Vanilla JS over React** | No build step, minimal complexity for 4 pages |
| **JWT over sessions** | Stateless auth, easy to scale, works with REST APIs |

---

## ✅ Requirements Compliance Matrix

| FR # | Requirement | Status | Implementation |
|------|------------|--------|----------------|
| **FR1** | Secure Register / Login / Logout | ✅ Done | `app/auth.py` — JWT + bcrypt |
| **FR3** | Accept JPEG uploads | ✅ Done | `app/upload.py` + `config.py` |
| **FR4** | Accept PNG uploads | ✅ Done | `app/upload.py` + `config.py` |
| **FR5** | Accept BMP uploads | ✅ Done | `app/upload.py` + `config.py` |
| **FR6** | Detect if image contains a battery (Yes/No) | ✅ Done | `ml/detector.py` — MobileNetV2 CNN |
| **FR7** | Check if battery text is visible and aligned | ✅ Done | `ml/aligner.py` — OpenCV Hough Lines |
| **FR8** | Extract brand, chemistry, voltage via OCR | ✅ Done | `ml/ocr_extractor.py` — Tesseract |
| **FR9** | Unified Python pipeline for all 3 stages | ✅ Done | `app/pipeline.py` — `/api/analyze` |
| **FR10** | Generate recyclable elements + recovery % | ✅ Done | `app/eda.py` + `materials_db.json` |
| **FR11** | Process different battery types accordingly | ✅ Done | 6 chemistries supported end-to-end |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                           │
│  login.html  ·  register.html  ·  dashboard.html  ·  results.html  │
│           (Professional Light UI — Vanilla HTML/CSS/JS)              │
└────────────────────────────┬────────────────────────────────────────┘
                             │ HTTP / REST API
┌────────────────────────────▼────────────────────────────────────────┐
│                         API LAYER (Flask)                           │
│  /api/register  /api/login  /api/logout  /api/me                   │
│  /api/upload    /api/analyze   /api/recover   /api/chemistries      │
│  JWT Authentication  ·  Flask-CORS  ·  File Validation (16 MB)     │
└──────────┬────────────────────┬─────────────────┬───────────────────┘
           │                    │                 │
┌──────────▼──────────┐  ┌──────▼─────────┐  ┌───▼────────────────────┐
│    AUTH & USERS     │  │   CV PIPELINE  │  │    EDA / MATERIALS     │
│  Flask-SQLAlchemy   │  │                │  │                        │
│  SQLite Database    │  │  Stage 1 ─────▶ │  │  materials_db.json     │
│  bcrypt password    │  │  BatteryDetect  │  │  6 chemistries         │
│  JWT blocklist      │  │  Stage 2 ─────▶ │  │  25+ elements          │
│  (in-memory set)    │  │  TextAligner    │  │  min/max/avg/notes     │
└─────────────────────┘  │  Stage 3 ─────▶ │  └────────────────────────┘
                         │  OCR Extractor  │
                         └────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────┐
│                          ML LAYER                                   │
│  MobileNetV2 (fine-tuned)   │  OpenCV heuristics   │  Tesseract OCR │
│  SLIBR dataset (1,510 imgs) │  Hough transforms    │  Multi-PSM     │
│  99.67% Val Accuracy        │  Edge density        │  Regex extract │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Machine Learning Details

### Model: MobileNetV2 (Fine-Tuned)

| Property | Value |
|----------|-------|
| **Base Architecture** | MobileNetV2 (ImageNet pre-trained) |
| **Classifier Head** | Linear layer → 3 output classes |
| **Dataset** | SLIBR (Spent Lithium-Ion Battery Recycling) |
| **Dataset Size** | 1,510 labelled images |
| **Classes** | `anode`, `cathode`, `nothing` |
| **Training Strategy** | 2-phase (frozen backbone → unfrozen fine-tuning) |
| **Epochs** | 25 |
| **Optimizer** | AdamW (weight_decay = 1e-4) |
| **Scheduler** | Cosine Annealing LR |
| **Best Val Accuracy** | **99.67%** (achieved at Epoch 21) |
| **Model File** | `models/battery_detector.pth` |
| **Augmentation** | Random crop, flip, rotation, color jitter |

### Training Progression

| Epoch | Val Accuracy | Notes |
|-------|-------------|-------|
| 1 | 80.46% | Frozen backbone |
| 5 | 91.06% | Frozen backbone |
| 10 | 92.72% | Frozen backbone |
| **12** | **97.02%** | 🔓 Backbone unfrozen |
| 14 | 99.01% | New best |
| **21** | **99.67%** | ⭐ Best model saved |
| 25 | 98.34% | Training complete |

### Computer Vision Pipeline Details

```
Image → (JPEG / PNG / BMP up to 16 MB)
  │
  ▼
Stage 1: Battery Detection (ml/detector.py)
  ├── MobileNetV2 CNN (if models/battery_detector.pth exists)
  │     └── Softmax probabilities → battery_class (anode/cathode/nothing)
  └── Fallback: OpenCV heuristic
        ├── Hough Circle Transform (cylindrical batteries)
        ├── Edge density → Canny edges
        └── Rectangular contour aspect ratio check
  → Output: { battery_detected: bool, confidence: float }
  │
  ▼ (only if battery_detected = true)
Stage 2: Text Visibility & Alignment (ml/aligner.py)
  ├── Contrast check: std > 25, 20 < mean_brightness < 235
  ├── Edge density: Canny edges > 4% of pixels = text visible
  └── Hough Line Transform: median angle deviation ≤ ±15° = aligned
  → Output: { text_visible: bool, aligned: bool, angle: float }
  │
  ▼
Stage 3: OCR Extraction (ml/ocr_extractor.py)
  ├── Pre-processing: resize, sharpen, 2× contrast boost (PIL)
  ├── Tesseract OCR — 3 PSM modes (3, 6, 11)
  ├── Brand: lexicon match (25+ known brands)
  ├── Chemistry: regex match (6 chemistries, 30+ patterns)
  └── Voltage: regex pattern (e.g. "3.7V", "12V", "1.5 volts")
  → Output: { brand, chemistry, voltage, raw_text }
  │
  ▼
EDA Page → Auto-loads recovery % for detected chemistry
```

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Name** | Spent Lithium-Ion Battery Recycling (SLIBR) |
| **Source** | [Kaggle — thgere/spent-lithium-ion-battery-recyclingslibr-dataset](https://www.kaggle.com/datasets/thgere/spent-lithium-ion-battery-recyclingslibr-dataset/data) |
| **Total Images** | 1,510 |
| **Classes** | `anode` · `cathode` · `nothing` |
| **Sub-datasets** | Classification (used for training) + Object Detection (labels available) |
| **Train/Val Split** | 80% training / 20% validation (random split) |
| **Validation Size** | ~302 images |
| **Image Format** | JPEG |
| **Download Size** | ~1.9 GB |

To download with Python:
```python
import kagglehub
path = kagglehub.dataset_download("thgere/spent-lithium-ion-battery-recyclingslibr-dataset")
print("Path to dataset files:", path)
```

---

## 🔋 Supported Battery Chemistries & Recovery Data

### Lithium-Ion (Li-ion)
| Element | Symbol | Min % | Max % | Notes |
|---------|--------|-------|-------|-------|
| Cobalt | Co | 90% | 98% | High-value cathode material; hydrometallurgical recovery |
| Nickel | Ni | 85% | 95% | Recovered via solvent extraction |
| Lithium | Li | 70% | 99% | Modern specialized processes exceed 99% |
| Copper | Cu | 90% | 98% | Current collector foils; efficient mechanical separation |
| Aluminum | Al | 90% | 97% | Cathode current collector |
| Graphite | C | 60% | 85% | Anode material; reused in new anode production |
| Manganese | Mn | 75% | 92% | Present in NMC and LMO chemistries |

### Lithium Iron Phosphate (LiFePO4)
| Element | Symbol | Min % | Max % |
|---------|--------|-------|-------|
| Lithium | Li | 75% | 95% |
| Iron | Fe | 85% | 97% |
| Phosphate | PO4 | 80% | 95% |
| Copper | Cu | 88% | 97% |
| Aluminum | Al | 88% | 96% |
| Carbon (Graphite) | C | 55% | 80% |

### Other Supported Chemistries
| Chemistry | Key Elements | Highest Recovery |
|-----------|-------------|-----------------|
| **NiMH** | Ni, Co, Rare Earth (La/Ce/Nd), Fe, Zn | Ni: 90–98% |
| **NiCd** | Cd ⚠️ (TOXIC), Ni, Steel, KOH | Cd: 90–99% (mandatory) |
| **Lead-Acid** | Pb, H₂SO₄, Polypropylene, Sb | Pb: 95–99% |
| **Alkaline** | Zn, MnO₂, Steel, KOH | Steel: 85–95% |

---

## 📁 Project Structure

```
battery_recycling_system/
│
├── app/                          # Flask Application
│   ├── __init__.py               # App factory, extensions, blueprint registration
│   ├── auth.py                   # FR1: Register / Login / Logout (JWT)
│   ├── upload.py                 # FR3–FR5: Image upload (JPEG/PNG/BMP)
│   ├── pipeline.py               # FR9: Unified 3-stage analysis pipeline
│   ├── eda.py                    # FR10–FR11: Recovery percentage API
│   ├── models.py                 # SQLAlchemy User model
│   ├── views.py                  # HTML page routes (login, dashboard, results)
│   └── materials_db.json         # Battery chemistry → elements → recovery %
│
├── ml/                           # Machine Learning
│   ├── detector.py               # FR6: Battery detection (CNN + heuristic)
│   ├── aligner.py                # FR7: Text visibility & alignment (OpenCV)
│   ├── ocr_extractor.py          # FR8: OCR extraction (Tesseract + regex)
│   └── train_model.py            # MobileNetV2 fine-tuning script (SLIBR)
│
├── frontend/
│   ├── static/css/style.css      # Professional light UI
│   └── templates/
│       ├── login.html            # Login page
│       ├── register.html         # Registration page
│       ├── dashboard.html        # Upload + 3-stage pipeline view
│       └── results.html          # EDA: recovery charts + data tables
│
├── models/
│   ├── battery_detector.pth      # ✅ Trained model weights (99.67% accuracy)
│   └── training_history_*.json   # Training metrics log
│
├── uploads/                      # Temporarily stored uploaded images
├── data/raw/                     # Training dataset directory
├── venv/                         # Python virtual environment
├── config.py                     # All configuration (JWT, DB, paths, limits)
├── run.py                        # App entry point
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## ⚙️ Setup & Installation

### Prerequisites

| Tool | Version | Required | Purpose |
|------|---------|----------|---------|
| **Python** | 3.9 – 3.14 | ✅ Yes | Core runtime |
| **pip** | 21+ | ✅ Yes | Python package manager (ships with Python) |
| **Tesseract OCR** | 4.x – 5.x | ✅ Yes | OCR engine for Stage 3 text extraction |
| **Git** | Any | ✅ Yes | Clone the repository |
| **Homebrew** | Any | macOS only | Install Tesseract & Python on macOS |

### System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **RAM** | 4 GB | 8 GB |
| **Disk Space** | ~500 MB (code + deps) | ~2.5 GB (with dataset) |
| **CPU** | Any x86_64 / ARM64 | Multi-core for faster inference |
| **GPU** | Not required | CUDA GPU optional for training |
| **OS** | macOS 11+, Windows 10+, Ubuntu 20.04+ | — |

---

### 🍎 macOS Installation (Intel & Apple Silicon)

#### 1. Install Homebrew (if not installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

After install, follow the instructions Homebrew prints to add it to your PATH.

#### 2. Install Python & Tesseract

```bash
brew install python@3.12 tesseract
```

Verify:
```bash
python3 --version     # Python 3.12.x
tesseract --version   # tesseract 5.x.x
```

**Tesseract paths** (auto-detected by the app):
| Mac Type | Path |
|----------|------|
| Apple Silicon (M1/M2/M3/M4) | `/opt/homebrew/bin/tesseract` |
| Intel Mac | `/usr/local/bin/tesseract` |

#### 3. Clone & Set Up

```bash
git clone https://github.com/srivardhan-kondu/AI-Battery-.git
cd AI-Battery-

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

> **Apple Silicon note**: PyTorch ships native ARM wheels — `pip install -r requirements.txt` works directly. If you hit errors:
> ```bash
> pip install torch torchvision
> ```

#### 4. Run

```bash
python run.py
```

Open **http://localhost:5000** in your browser.

> **⚠️ Port 5000 conflict** — macOS Monterey+ uses port 5000 for AirPlay Receiver (Control Center → AirDrop & Handoff). Fix:
> ```bash
> # Option A: Disable AirPlay Receiver in System Settings
> # Option B: Use port 5001 instead
> python -c "
> from app import create_app
> app = create_app()
> app.run(host='0.0.0.0', port=5001, debug=True)
> "
> ```

---

### 🪟 Windows Installation (Windows 10 / 11)

#### 1. Install Python

1. Download Python 3.12 from [python.org/downloads](https://www.python.org/downloads/)
2. **IMPORTANT**: Check ☑ **"Add Python to PATH"** during installation
3. Verify in **Command Prompt** or **PowerShell**:
   ```cmd
   python --version
   pip --version
   ```

#### 2. Install Tesseract OCR

1. Download the Windows installer from:
   **https://github.com/UB-Mannheim/tesseract/wiki**
   (Download the `.exe` — typically `tesseract-ocr-w64-setup-5.x.x.exe`)
2. Run the installer — **note the install path** (default: `C:\Program Files\Tesseract-OCR`)
3. **Add Tesseract to your system PATH**:
   - Open **Start** → search **"Environment Variables"** → click **"Edit the system environment variables"**
   - Click **"Environment Variables…"**
   - Under **System Variables**, select **Path** → click **Edit**
   - Click **New** → paste: `C:\Program Files\Tesseract-OCR`
   - Click **OK** on all dialogs
4. **Restart** your Command Prompt / PowerShell
5. Verify:
   ```cmd
   tesseract --version
   ```

> **Alternative**: If you don't want to modify PATH, set an environment variable:
> ```cmd
> set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
> ```
> Or the app will auto-detect the default install path.

#### 3. Install Git

Download from [git-scm.com](https://git-scm.com/download/win) and install. Select **"Git from the command line"** during setup.

#### 4. Clone & Set Up

**Command Prompt:**
```cmd
git clone https://github.com/srivardhan-kondu/AI-Battery-.git
cd AI-Battery-

python -m venv venv
venv\Scripts\activate

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**PowerShell:**
```powershell
git clone https://github.com/srivardhan-kondu/AI-Battery-.git
cd AI-Battery-

python -m venv venv
venv\Scripts\Activate.ps1

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

> **PowerShell execution policy error?** Run this first:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

> **PyTorch install issue on Windows?** Install CPU-only build:
> ```cmd
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```

> **OpenCV `ImportError: DLL load failed`?** Install the Visual C++ Redistributable:
> Download from [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) and restart.

#### 5. Run

```cmd
python run.py
```

Open **http://localhost:5000** in your browser.

---

### 🐧 Linux Installation (Ubuntu / Debian / Fedora / Arch)

#### Ubuntu / Debian

```bash
# 1. Install Python, pip, venv, Tesseract, and OpenCV system deps
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv \
  tesseract-ocr tesseract-ocr-eng \
  libgl1-mesa-glx libglib2.0-0

# 2. Verify
python3 --version
tesseract --version
```

#### Fedora / RHEL / CentOS

```bash
sudo dnf install -y python3 python3-pip python3-virtualenv \
  tesseract tesseract-langpack-eng \
  mesa-libGL glib2

python3 --version
tesseract --version
```

#### Arch Linux

```bash
sudo pacman -S python python-pip tesseract tesseract-data-eng
```

#### Clone & Set Up (all distros)

```bash
git clone https://github.com/srivardhan-kondu/AI-Battery-.git
cd AI-Battery-

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

> **`libGL` / OpenCV error?** This is the most common Linux issue:
> ```bash
> # Ubuntu/Debian
> sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
> # Or use headless OpenCV:
> pip uninstall opencv-python && pip install opencv-python-headless
> ```

> **PyTorch CPU-only (recommended for servers without GPU):**
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```

#### Run

```bash
python run.py
```

Open **http://localhost:5000**.

---

### ✅ Post-Install Verification (All Platforms)

Run this one-liner to confirm every dependency is correctly installed:

```bash
python -c "
import flask; print(f'Flask:          {flask.__version__}')
import torch; print(f'PyTorch:        {torch.__version__}')
import cv2;   print(f'OpenCV:         {cv2.__version__}')
import PIL;   print(f'Pillow:         {PIL.__version__}')
import pytesseract; print(f'pytesseract:    OK')
import numpy; print(f'NumPy:          {numpy.__version__}')
import pandas; print(f'Pandas:         {pandas.__version__}')
import sklearn; print(f'scikit-learn:   {sklearn.__version__}')
print()
print('All dependencies installed successfully!')
"
```

Expected output:
```
Flask:          3.0.2
PyTorch:        2.2.1
OpenCV:         4.9.0
Pillow:         10.2.0
pytesseract:    OK
NumPy:          1.26.4
Pandas:         2.2.1
scikit-learn:   1.4.1
All dependencies installed successfully!
```

---

### 📝 Register & First Test

1. Open **http://localhost:5000** (or `:5001` on macOS if port 5000 is busy)
2. Click **Register** and create an account:
   - **Email**: any valid email (e.g., `test@batteryai.com`)
   - **Password**: minimum 8 characters, with 1 uppercase, 1 lowercase, 1 digit (e.g., `Test@1234`)
3. Upload a battery image (JPEG, PNG, or BMP, up to 16 MB)
4. View the 3-stage analysis results on the Dashboard
5. Click **View Recovery Analysis** to see EDA — recoverable elements, hazard levels, and recovery percentages
6. Click **⬇ Download Report** to export the results as **PDF** or **CSV**

---

### 🔧 Troubleshooting Guide

<details>
<summary><strong>Click to expand all common issues & fixes</strong></summary>

#### Tesseract Issues

| Problem | Platform | Solution |
|---------|----------|----------|
| `TesseractNotFoundError` | All | Ensure Tesseract is installed and on PATH. Run `tesseract --version` to verify. |
| OCR returns all "Unknown" | All | Tesseract isn't at the expected path. Set `TESSERACT_CMD` env variable to your install path. |
| `tesseract: command not found` | Linux | `sudo apt install tesseract-ocr` |
| `'tesseract' is not recognized` | Windows | Add `C:\Program Files\Tesseract-OCR` to your system PATH (see Windows setup above). |

#### Python / pip Issues

| Problem | Platform | Solution |
|---------|----------|----------|
| `python: command not found` | macOS/Linux | Use `python3` instead of `python` |
| `python3` not found | Windows | Reinstall Python with "Add to PATH" checked |
| `pip install` hangs on PyTorch | All | Install CPU-only: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` |
| `ModuleNotFoundError: No module named 'cv2'` | All | `pip install opencv-python` |
| Pillow build error on Python 3.13+ | All | `pip install Pillow --upgrade` |

#### OpenCV Issues

| Problem | Platform | Solution |
|---------|----------|----------|
| `ImportError: libGL.so.1` | Linux | `sudo apt install libgl1-mesa-glx` or use `pip install opencv-python-headless` |
| `ImportError: DLL load failed` | Windows | Install [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) |
| `ImportError: libgthread` | Linux | `sudo apt install libglib2.0-0` |

#### Server Issues

| Problem | Platform | Solution |
|---------|----------|----------|
| Port 5000 already in use | macOS | AirPlay Receiver uses 5000. Disable it or use port 5001 (see macOS section above). |
| Port 5000 already in use | Any | `lsof -ti:5000 \| xargs kill -9` (macOS/Linux) or find/kill in Task Manager (Windows) |
| `Address already in use` | Any | Wait 30 seconds (TIME_WAIT) or kill the old process |
| CORS errors in browser | Any | Ensure `Flask-CORS` is installed: `pip install flask-cors` |

#### PowerShell Issues (Windows)

| Problem | Solution |
|---------|----------|
| `venv\Scripts\Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| `pip` not recognized after venv activation | Use `python -m pip install` instead |

</details>

---

### 📦 All Python Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| `Flask` | 3.0.2 | Web framework |
| `Flask-SQLAlchemy` | 3.1.1 | Database ORM |
| `Flask-JWT-Extended` | 4.6.0 | JWT authentication |
| `Flask-Bcrypt` | 1.0.1 | Password hashing |
| `Flask-CORS` | 4.0.0 | Cross-origin requests |
| `Pillow` | 10.2.0+ | Image processing |
| `opencv-python` | 4.9.0+ | Computer vision (Stages 1 & 2) |
| `pytesseract` | 0.3.10 | Tesseract OCR binding (Stage 3) |
| `torch` | 2.2.1+ | Deep learning framework |
| `torchvision` | 0.17.1+ | Pre-trained models & transforms |
| `ultralytics` | 8.1.28+ | YOLO utilities |
| `numpy` | 1.26.4+ | Numerical computing |
| `pandas` | 2.2.1+ | Data handling |
| `scikit-learn` | 1.4.1+ | ML utilities |
| `matplotlib` | 3.8.3+ | Plotting (training) |
| `seaborn` | 0.13.2+ | Statistical visualization |
| `requests` | 2.31.0+ | HTTP client |
| `python-dotenv` | 1.0.1 | Environment variables from `.env` |
| `Werkzeug` | 3.0.1+ | WSGI utilities (secure uploads) |
| `SQLAlchemy` | 2.0.27+ | SQL toolkit |
| `tqdm` | 4.66.2+ | Progress bars (training) |

### System Dependencies (Non-Python)

| Dependency | macOS | Windows | Ubuntu/Debian | Fedora |
|-----------|-------|---------|---------------|--------|
| **Python 3.9+** | `brew install python@3.12` | [python.org](https://python.org) | `sudo apt install python3` | `sudo dnf install python3` |
| **Tesseract 4+** | `brew install tesseract` | [UB-Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki) | `sudo apt install tesseract-ocr` | `sudo dnf install tesseract` |
| **Git** | `brew install git` | [git-scm.com](https://git-scm.com) | `sudo apt install git` | `sudo dnf install git` |
| **libGL** (OpenCV) | ✅ Built-in | ✅ Built-in | `sudo apt install libgl1-mesa-glx` | `sudo dnf install mesa-libGL` |
| **VC++ Runtime** | N/A | [Download](https://aka.ms/vs/17/release/vc_redist.x64.exe) | N/A | N/A |

---

### ⚙️ Environment Variables (Optional)

The app works out of the box with defaults. Override any setting via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | `battery-recycling-super-secret-key-2024` | Flask session secret |
| `JWT_SECRET_KEY` | `jwt-battery-secret-2024` | JWT signing key |
| `DATABASE_URL` | `sqlite:///battery_recycling.db` | SQLAlchemy database URI |
| `TESSERACT_CMD` | Auto-detected | Full path to Tesseract binary |
| `DEBUG` | `True` | Enable Flask debug mode |

Set via terminal:
```bash
# macOS / Linux
export TESSERACT_CMD="/usr/local/bin/tesseract"
export SECRET_KEY="your-production-secret"

# Windows (CMD)
set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
set SECRET_KEY=your-production-secret

# Windows (PowerShell)
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
$env:SECRET_KEY = "your-production-secret"
```

Or create a `.env` file in the project root (loaded by `python-dotenv`):
```env
SECRET_KEY=your-production-secret
JWT_SECRET_KEY=your-jwt-secret
TESSERACT_CMD=/usr/local/bin/tesseract
DEBUG=False
```

---

### 🔄 Cross-Platform Compatibility Notes

| Feature | macOS | Windows | Linux | Notes |
|---------|-------|---------|-------|-------|
| Tesseract detection | ✅ Auto | ✅ Auto | ✅ Auto | `config.py` checks `PATH`, then platform-specific paths |
| SQLite database | ✅ | ✅ | ✅ | Path backslashes normalized for Windows |
| File uploads | ✅ | ✅ | ✅ | `os.path.join()` everywhere, no hardcoded separators |
| Port 5000 | ⚠️ AirPlay | ✅ | ✅ | Use port 5001 on macOS Monterey+ |
| Python command | `python3` | `python` | `python3` | Some Linux distros only have `python3` |
| Virtual env activation | `source venv/bin/activate` | `venv\Scripts\activate` | `source venv/bin/activate` | — |
| OCR inference | ✅ | ✅ | ✅ | All 3 PSM modes work cross-platform |
| PDF/CSV export | ✅ | ✅ | ✅ | Client-side generation (jsPDF) — no server deps |

---

## 🧠 Model Training (Already Done — Skip if using pre-trained weights)

The trained model (`models/battery_detector.pth`) is already included. To retrain from scratch:

### Step 1 — Download the Dataset

```python
import kagglehub
path = kagglehub.dataset_download("thgere/spent-lithium-ion-battery-recyclingslibr-dataset")
print("Dataset path:", path)
# Expected: ~/.cache/kagglehub/datasets/thgere/.../versions/1/SLiBR_dataset/classification
```

### Step 2 — Run Training Script

```bash
source venv/bin/activate

python ml/train_model.py \
  --data_dir "/path/to/SLiBR_dataset/classification" \
  --epochs 25 \
  --batch_size 32 \
  --output_dir models
```

**Training Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `data/raw` | Path to dataset with class subfolders |
| `--epochs` | 25 | Number of training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--val_split` | 0.2 | Validation split ratio |
| `--output_dir` | `models` | Where to save weights |
| `--no_freeze` | False | Unfreeze backbone from epoch 1 |

---

## 🔌 API Reference

### Authentication

#### `POST /api/register`
```json
// Request
{ "email": "user@example.com", "password": "Password123" }

// Response 201
{ "message": "Registration successful", "token": "<JWT>", "user": { "id": 1, "email": "..." } }
```

#### `POST /api/login`
```json
// Request
{ "email": "user@example.com", "password": "Password123" }

// Response 200
{ "message": "Login successful", "token": "<JWT>", "user": { "id": 1, "email": "..." } }
```

#### `POST /api/logout`
```
Authorization: Bearer <JWT>
// Response 200: { "message": "Successfully logged out" }
```

---

### Image Upload

#### `POST /api/upload`
```
Authorization: Bearer <JWT>
Content-Type: multipart/form-data

file: <image.jpg | image.png | image.bmp>

// Response 200
{
  "message": "File uploaded successfully",
  "filename": "a3f9e1d2.jpg",
  "format": "JPEG"
}
```

---

### Full Analysis Pipeline

#### `POST /api/analyze`
```
Authorization: Bearer <JWT>
Content-Type: multipart/form-data

file: <battery_image.jpg>
```

```json
// Response 200
{
  "filename": "a3f9e1d2.jpg",
  "stage_1_detection": {
    "battery_detected": true,
    "confidence": 0.9934,
    "method": "cnn_mobilenetv2"
  },
  "stage_2_alignment": {
    "text_visible": true,
    "aligned": true,
    "angle": -2.5,
    "contrast_score": 0.712,
    "edge_density": 0.0821,
    "method": "opencv_hough"
  },
  "stage_3_ocr": {
    "brand": "Panasonic",
    "chemistry": "NiMH",
    "voltage": "1.2V",
    "tesseract_available": true
  },
  "summary": {
    "battery_detected": true,
    "text_visible": true,
    "text_aligned": true,
    "brand": "Panasonic",
    "chemistry": "NiMH",
    "voltage": "1.2V",
    "confidence": 0.9934
  }
}
```

---

### EDA — Recovery Analysis

#### `POST /api/recover`
```json
// Request
{ "chemistry": "Li-ion", "voltage": "3.7V" }

// Response 200
{
  "chemistry": "Li-ion",
  "full_name": "Lithium-Ion",
  "typical_voltage": "3.6-3.7V",
  "common_brands": ["LG", "Samsung", "Panasonic"],
  "element_count": 7,
  "chart_data": {
    "labels": ["Cobalt", "Nickel", "Lithium", "Copper", "Aluminum", "Graphite", "Manganese"],
    "symbols": ["Co", "Ni", "Li", "Cu", "Al", "C", "Mn"],
    "min_values": [90, 85, 70, 90, 90, 60, 75],
    "max_values": [98, 95, 99, 98, 97, 85, 92],
    "avg_values": [94.0, 90.0, 84.5, 94.0, 93.5, 72.5, 83.5],
    "notes": ["High-value cathode material...", "..."],
    "colors": ["#00d4ff", "#7b2ff7", "#00ff88", ...]
  },
  "raw_elements": { ... }
}
```

#### `GET /api/chemistries`
```json
// Response 200
{
  "chemistries": [
    { "key": "Li-ion", "full_name": "Lithium-Ion", "typical_voltage": "3.6-3.7V", "element_count": 7 },
    { "key": "LiFePO4", "full_name": "Lithium Iron Phosphate", "typical_voltage": "3.2V", "element_count": 6 },
    { "key": "NiMH", "full_name": "Nickel-Metal Hydride", "typical_voltage": "1.2V", "element_count": 5 },
    { "key": "NiCd", "full_name": "Nickel-Cadmium", "typical_voltage": "1.2V", "element_count": 4 },
    { "key": "Lead-Acid", "full_name": "Lead-Acid", "typical_voltage": "2.0V (12V nominal)", "element_count": 4 },
    { "key": "Alkaline", "full_name": "Alkaline (Zinc-Manganese Dioxide)", "typical_voltage": "1.5V", "element_count": 4 }
  ]
}
```

---

## 🧪 Testing

### Test Credentials (Development)

Use these credentials to test the application without creating a new account:

| Field | Value |
|-------|-------|
| Email | `test@batteryai.com` |
| Password | `Test@1234` |

> **Note**: On first run, register with any email matching the password policy: ≥8 chars, 1 uppercase, 1 lowercase, 1 digit.

---

### API Testing with cURL

#### Register
```bash
curl -X POST http://localhost:5000/api/register \
  -H "Content-Type: application/json" \
  -d '{"email": "test@batteryai.com", "password": "Test@1234"}'
```

#### Login & Save Token
```bash
TOKEN=$(curl -s -X POST http://localhost:5000/api/login \
  -H "Content-Type: application/json" \
  -d '{"email": "test@batteryai.com", "password": "Test@1234"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['token'])")
echo "JWT: $TOKEN"
```

#### Analyze a Battery Image
```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/path/to/your/battery_image.jpg"
```

#### Get Recovery Data for Li-ion
```bash
curl -X POST http://localhost:5000/api/recover \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"chemistry": "Li-ion", "voltage": "3.7V"}'
```

#### List All Supported Chemistries
```bash
curl http://localhost:5000/api/chemistries
```

---

### Test Battery Images

Use these publicly available images to test the pipeline:

| Battery Type | Suggested Test Image | Expected Chemistry |
|-------------|---------------------|-------------------|
| Li-ion | AA lithium-ion cell (e.g., 18650) | `Li-ion` |
| NiMH | Panasonic Eneloop AA | `NiMH` |
| Alkaline | Duracell AA / AAA | `Alkaline` |
| Lead-Acid | Car battery / VRLA | `Lead-Acid` |

> **Tip**: For OCR best results, use clear, high-resolution images (≥400×400 px) where battery label text is visible and horizontal.

---

### Pipeline Validation Checklist

Run the following checks to validate the full system:

- [ ] `GET /` → Redirects to Login page
- [ ] `POST /api/register` → Returns JWT token
- [ ] `POST /api/login` → Returns JWT token
- [ ] `GET /api/chemistries` → Returns 6 battery types
- [ ] `POST /api/analyze` + battery image → Returns `battery_detected: true`
- [ ] `POST /api/analyze` + non-battery image → Returns `battery_detected: false`
- [ ] `POST /api/recover` + `{"chemistry": "Li-ion"}` → Returns 7 elements
- [ ] `POST /api/logout` → Invalidates token
- [ ] `POST /api/analyze` with invalidated token → Returns `401 Unauthorized`

---

## 🛡️ Security Features

| Feature | Implementation |
|---------|---------------|
| Password Hashing | `bcrypt` (cost factor 12) |
| Authentication Tokens | JWT with 24-hour expiry |
| Token Invalidation | Server-side blocklist (`set` in memory) |
| File Type Validation | Extension + MIME whitelist |
| Upload Size Limit | 16 MB maximum |
| CORS | `flask-cors` with credentials support |
| Protected Routes | `@jwt_required()` on all sensitive endpoints |
| SQL Injection | SQLAlchemy ORM (parameterized queries) |

---

## 📦 Dependencies

```
flask>=3.0.0
flask-sqlalchemy>=3.1.0
flask-jwt-extended>=4.6.0
flask-bcrypt>=1.0.1
flask-cors>=4.0.0
torch>=2.0.0
torchvision>=0.15.0
Pillow>=10.0.0
opencv-python>=4.8.0
pytesseract>=0.3.10
numpy>=1.24.0
kagglehub>=0.3.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🗺️ User Flow

```
[Browser] ──▶ / (Login Page)
                │
                ├── Register ──▶ POST /api/register ──▶ JWT Token stored
                └── Login    ──▶ POST /api/login    ──▶ JWT Token stored
                                         │
                                         ▼
                             /dashboard (Upload Page)
                                         │
                             Drop / Select battery image
                             (JPEG / PNG / BMP ≤ 16 MB)
                                         │
                                         ▼
                             POST /api/analyze
                             ┌────────────────────┐
                             │ Stage 1: Detect    │ ──▶ battery_detected: true/false
                             │ Stage 2: Align     │ ──▶ text_visible, angle
                             │ Stage 3: OCR       │ ──▶ brand, chemistry, voltage
                             └────────────────────┘
                                         │
                             Click "View Recovery Analysis"
                                         │
                                         ▼
                             /results (EDA Page)
                             POST /api/recover
                             ──▶ Bar chart: min/max/avg per element
                             ──▶ Data table with notes
                             ──▶ Auto-loads detected chemistry
```

---

## 📈 Performance Summary

| Metric | Value |
|--------|-------|
| Model Architecture | MobileNetV2 |
| Best Validation Accuracy | **99.67%** |
| Training Dataset | SLIBR (1,510 images, 3 classes) |
| Training Duration | ~1 hour (CPU — Apple Silicon M-series) |
| Inference Time (CPU) | ~150–300 ms per image |
| Supported Upload Formats | JPEG, PNG, BMP |
| Max Upload Size | 16 MB |
| Supported Battery Chemistries | 6 |
| Total Recyclable Elements Tracked | 25+ |

---

## 🔮 Future Improvements

- [ ] Replace in-memory JWT blocklist with Redis for production scalability
- [ ] Add object detection head using the SLIBR YOLO labels for bounding box output
- [ ] Support video stream input for conveyor belt real-time classification
- [ ] Integrate weight/rating field to calculate total recoverable mass
- [x] ~~Export EDA results as PDF report~~ ✅ Implemented — PDF & CSV download
- [x] ~~Cross-platform compatibility~~ ✅ Mac, Windows, Linux all supported
- [ ] Docker containerization
- [ ] GPU inference support

---

## 📄 License

This project is developed for academic research purposes under the **MIT License**.

---

## 🙏 Acknowledgements

- **Dataset**: [SLIBR Dataset by thgere on Kaggle](https://www.kaggle.com/datasets/thgere/spent-lithium-ion-battery-recyclingslibr-dataset/data)
- **Base Model**: [MobileNetV2 — Sandler et al., 2018](https://arxiv.org/abs/1801.04381)
- **OCR Engine**: [Tesseract OCR — Google](https://github.com/tesseract-ocr/tesseract)
- **Framework**: [Flask — Pallets Projects](https://flask.palletsprojects.com/)
- **Deep Learning**: [PyTorch](https://pytorch.org/)

---

*Built with ❤️ for a greener planet — AI-powered battery recycling for a sustainable circular economy.*
