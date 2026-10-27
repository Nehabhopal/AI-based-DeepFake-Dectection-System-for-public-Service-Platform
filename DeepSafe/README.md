
# DeepSafe — AI-Based DeepFake Detection and Forensic Evidence System

DeepSafe is a source-code project for image-based deepfake classification using a trained PyTorch ResNet18 model, Grad-CAM explanations, Flask API, and MySQL persistence. The UI follows the supplied navy/blue public-service dashboard reference.

> **Important honesty notice:** no labeled image dataset or trained checkpoint was supplied with the reference image/specification. This ZIP does not contain a fabricated model, prediction, heatmap, or evaluation metric. Analysis requests return a clear error until you train a model and configure MySQL. The training/evaluation pipeline is included.

## Features
- Signup/login with password hashing and Flask session authentication.
- Image upload validation (JPG/JPEG/PNG/WEBP, MIME sniffing, size cap, safe generated names).
- ResNet18 training pipeline for REAL and DEEPFAKE image folders.
- Inference with calibrated-by-softmax class probabilities (not a guarantee of real-world calibration).
- Grad-CAM generated from the trained model's final convolutional block.
- Image quality summary and OpenCV face detection status (face detection is informational, not a requirement for classification).
- MySQL-backed history, feedback/report submissions, and admin summary.
- Responsive navy sidebar / light content UI.
- Automated tests for core utilities and API behavior.

## Architecture
Browser (HTML/CSS/JS) → Flask API → validation → trained ResNet18 inference → Grad-CAM → MySQL result persistence → JSON/UI.

## Folder structure
```text
DeepSafe/
├── backend/                 Flask app, config, database and routes
├── frontend/                HTML, CSS, JS
├── ml/                      dataset loader, model, train/evaluate/predict/Grad-CAM
├── models/                  expected checkpoint and generated metadata
├── dataset/                 empty dataset layout guidance
├── database/                schema.sql
├── uploads/                 user uploads and generated explanations
├── outputs/evaluation/      generated evaluation artifacts
├── docs/                    API, model, Grad-CAM and reference notes
├── tests/                   pytest tests
├── requirements.txt
└── .env.example
```

## Requirements
- Windows 11, Python 3.11 recommended (PyTorch/torchvision support varies by Python version; use a supported version rather than assuming Python 3.14 compatibility).
- MySQL Server 8.x and a database user.
- A labeled dataset with both classes. CUDA is optional; CPU training works but may be slow.

## Windows setup
Open PowerShell in the extracted `DeepSafe` folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set a strong `SECRET_KEY`, `DATABASE_URL`, and `ADMIN_EMAIL`. Create the database and tables:

```powershell
mysql -u root -p
```

At the MySQL prompt:
```sql
CREATE DATABASE deepsafe CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
EXIT;
```

Then from PowerShell:
```powershell
Get-Content .\database\schema.sql | mysql -u root -p deepsafe
```

Example SQLAlchemy URL (URL-encode special characters in the password):
`mysql+pymysql://deepsafe_user:your_password@127.0.0.1:3306/deepsafe`

Create an admin account by registering normally, then set its role to `admin` in MySQL:
```sql
UPDATE users SET role='admin' WHERE email='your-admin-email@example.com';
```

## Dataset layout
Place **real labeled images** here. Do not put the same image in multiple splits.

```text
dataset/
├── train/
│   ├── REAL/
│   └── DEEPFAKE/
├── val/
│   ├── REAL/
│   └── DEEPFAKE/
└── test/
    ├── REAL/
    └── DEEPFAKE/
```

The project does not redistribute a dataset. Choose a dataset you are permitted to use (for example, a research dataset with a license suitable for your project), retain its license/citation, and document its source. Keep identities and source videos split consistently where applicable to reduce leakage. The training loader supports common image formats.

## Train and evaluate
```powershell
python ml/train.py
python ml/evaluate.py
```
Training writes `models/deepfake_resnet18_best.pth`, `models/model_metadata.json`, and `outputs/evaluation/training_history.json`. Evaluation writes `outputs/evaluation/test_metrics.json` and a confusion-matrix image. Metrics are produced only after actual evaluation on your test split. Inspect class counts and split quality before interpreting results.

## Run the backend and frontend
```powershell
python backend/app.py
```
Open `http://127.0.0.1:5000/` in your browser. Flask serves the static frontend. MySQL must be running and configured. A missing database or model is surfaced as an error; the app does not silently fall back to fake predictions.

## API quick reference
- `GET /api/health` — service/model/database status.
- `POST /api/auth/signup` — JSON `{ "name": "...", "email": "...", "password": "..." }`.
- `POST /api/auth/login` — JSON `{ "email": "...", "password": "..." }`.
- `POST /api/auth/logout` — logout.
- `GET /api/auth/me` — current session user.
- `POST /api/analyze/image` — multipart field `image`, requires login.
- `GET /api/history` — current user's history, requires login.
- `GET /api/history/<id>` — owned analysis detail, requires login.
- `DELETE /api/history/<id>` — delete owned history item, requires login.
- `POST /api/feedback` — JSON `{ "category": "...", "message": "..." }`, requires login.
- `GET /api/admin/summary` — admin-only live database counts and model metadata.

## Grad-CAM and interpretation
Grad-CAM is calculated from gradients and activations of the trained ResNet18 `layer4` block. The resulting map is resized and overlaid on the uploaded image. It visualizes regions that influenced the selected model output; it is not proof of manipulation and can be misleading. See `docs/GRADCAM.md`.

## Security and privacy notes
- Use HTTPS, secure cookies, CSRF protection, rate limiting, and a proper production WSGI server before deployment.
- Local development uses Flask sessions and should not be exposed publicly as-is.
- Uploaded images may contain sensitive faces. Obtain consent, define retention/deletion policies, and avoid uploading private imagery without authorization.
- This is a research/educational aid, not forensic proof. Verify media using trusted independent sources.

## Troubleshooting
- **Model unavailable:** run `python ml/train.py` after placing a valid dataset in the expected folders.
- **Database unavailable:** check MySQL service, database name/user/password, and `.env` `DATABASE_URL`.
- **PowerShell activation blocked:** run `Set-ExecutionPolicy -Scope Process RemoteSigned` in the current PowerShell window, then activate again.
- **Torch install failure:** use a supported Python version and the official PyTorch installation selector for your OS/hardware.
- **No face detected:** face detection is an informational OpenCV Haar-cascade check; it does not prove an image is genuine or fake.
