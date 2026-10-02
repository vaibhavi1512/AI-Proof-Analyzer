# AI-Proof-Analyzer — MAYA / EVIDEX

**AI-Based Digital Evidence Authenticity Verification System and Legal Admissibility**

MAYA is the investigation platform. EVIDEX is the investigator interface. Together they support case work, evidence integrity, image analysis, and a production video authenticity pipeline with explainable outputs and forensic PDF reports.

Deepfake detection is one component of the product. A model score is not a legal finding.

## IMPORTANT FOR AI CODING ASSISTANTS

Read this README before modifying code.

- Preserve the final 16-frame LSTM video architecture.
- Preserve the separation between image analysis and video analysis.
- Preserve the no-face full-frame fallback. A missing face is not a FAKE decision.
- Do not modify model checkpoints casually.
- Do not commit secrets.
- Inspect the existing code before making changes.

## Current status

| Phase | Status |
|-------|--------|
| Phase 0 — Design | Complete (`docs/`) |
| Phase 1 — Foundation | Complete |
| Phase 2 — Evidence data engineering | Complete |
| Phase 2.5 — Dataset pipeline review and optimization | Complete |
| Phase 3.1 — AI model architecture | Complete |
| Phase 3.2 — AI training engine | Complete |
| Phase 3.3 — AI validation and reporting | Complete |
| Phase 3.4 — Investigation inference | Complete |
| Phase 3.5 — AI performance benchmarks | Complete |
| Phase 4.1 — Explainability (Grad-CAM foundation) | Complete |
| Phase 4.2 — Multi-explainer framework | Complete |
| Phase 4.3 — Explanation analytics and trust | Complete |
| Phase 4.4 — Explainability validation and benchmark | Complete |
| Phase 4.5 — Advanced explainability and trust layer | Complete |
| Phase 3 Product — Auth / cases / evidence / APIs | Complete |
| Phase 5 — Reports / hardening / Docker / image E2E | Complete |
| Phase 5 — Face reference verification (backend) | Complete |
| Phase 13 — Pre-manipulation analysis | Done |
| Phase 14 — Binary XAI | Done |
| Phase 15 — MAYA backend integration | Done |
| Phase 16 — EVIDEX frontend integration | Done |
| Phase 17 — End-to-end testing | Completed / verified |
| Application hardening | Done |
| Automatic report email | Done |

Future work that is **not** implemented: a manipulation-type classifier and a manipulation-type UI. The production video model is binary REAL / FAKE only.

Earlier design notes: [`docs/ROADMAP.md`](docs/ROADMAP.md)

---

## Production video pipeline

```
Video upload
  → video validation
  → deterministic 16-frame sampling
  → face-aware preprocessing
       face detected → face crop
       no face       → full-frame fallback
  → EfficientNet-B0
  → 1280-dimensional frame features
  → 2-layer LSTM, hidden size 256
  → REAL / FAKE
  → binary XAI
       relative temporal contribution
       Grad-CAM
  → EVIDEX
```

Production settings live in `ai/ffpp_video/constants.py`.

| Setting | Production value |
|---|---|
| Frames | 16 |
| Visual encoder | EfficientNet-B0 (`ffpp_visual_efficientnet_b0`) |
| Frame feature size | 1280 |
| Temporal model | 2-layer LSTM, hidden size 256 (`ffpp_video_lstm_v2`) |
| Model version | `v2-16frame` |
| Decision | FAKE when `p_fake` is at least 0.5, otherwise REAL |

16 frames are the final production configuration. LSTM is the final temporal model. A GRU was used only as a comparison experiment. A 32-frame setting was evaluated and is not the final configuration.

Every sampled frame is kept. If a face is not detected, that frame uses a full-frame fallback. No-face does not mean FAKE.

Source for this pipeline is `ai/ffpp_video/` and `ai/video/`. Image inference remains a separate EfficientNet-B0 path under `ai/inference/` and `ai/models/`.

### Final checkpoints

These two files are the production video inference checkpoints shipped with the repository:

```
artifacts/checkpoints/video/visual_model.pt
artifacts/checkpoints/video/lstm.pt
```

Leave `VIDEO_VISUAL_CHECKPOINT` and `VIDEO_LSTM_CHECKPOINT` unset to use those paths. Point them at another copy only when you intentionally want a different pair. Do not replace or retrain these files as part of ordinary application work.

### Final 16-frame LSTM test metrics

These figures are from the completed FaceForensics++ (FF++) test split used for this model. They are not universal deepfake-detection performance.

| Metric | Value |
|---|---:|
| Accuracy | 0.7000 |
| Precision | 0.8889 |
| Recall | 0.7143 |
| F1 | 0.7921 |
| Specificity | 0.6429 |
| Balanced accuracy | 0.6786 |
| ROC-AUC | 0.7666 |

Confusion matrix. Rows are the true class and columns are the predicted class, in the order REAL, FAKE:

```
[[9, 5],
 [16, 40]]
```

The first row is true REAL (9 correct, 5 called FAKE). The second row is true FAKE (16 called REAL, 40 correct). Those counts are the source of the metrics above: accuracy `(9 + 40) / 70 = 0.7000`.

The test split is a limited FF++ subset. It is not a stand-in for every real-world video.

### Face-aware preprocessing

Face-aware preprocessing selects a face crop when a face is detected and keeps the selection temporally continuous across the 16 frames. When no face is detected, the frame is still analyzed as a full-frame fallback.

A completed audit of the sampled frames counted:

| Item | Count |
|---|---:|
| Sampled frames | 8000 |
| Face-crop frames | 7957 |
| Full-frame fallback frames | 43 |
| Videos with at least one fallback | 21 |

The fallback subgroup is small. Those counts show that fallback happens and that those frames are retained. They do not support a strong claim about fallback-only accuracy.

---

## Confidence Score

For binary video classification, Confidence Score is the model probability of the predicted class:

- FAKE: Confidence Score = `p_fake`
- REAL: Confidence Score = `1 - p_fake`

EVIDEX shows that value as a percentage to two decimal places. It is a model-derived probability for the predicted class. It is not legal certainty, and it is not proof.

Image analysis keeps its own image Model Confidence display. Video screens do not replace that image field.

---

## Explainable analysis (binary video XAI)

Video XAI has two parts:

1. **Relative temporal contribution** — how the 16 frame positions contribute relative to each other.
2. **Grad-CAM spatial explanation** — regions contributing to the model prediction, including a video contact sheet and per-frame Grad-CAM artifacts.

Preferred wording:

- “Regions contributing to the model prediction”
- “Relative temporal contribution”

XAI explains model behavior. It does not prove manipulation, and it does not establish a legal conclusion.

Image explainability remains the separate Phase 4 stack under `ai/explainability/` (Grad-CAM and optional advanced explainers). Video screens do not use the image artifact routes.

---

## EVIDEX video screens

EVIDEX keeps three distinct views for one completed video analysis. They are not three copies of the same page.

### Analysis

- Prediction (REAL or FAKE)
- Confidence Score
- Model and model version
- Frames analyzed
- Face-crop frame count
- Full-frame fallback count
- Fallback note when a frame had no detected face
- Relative temporal contribution graph

The Analysis screen does not show the Grad-CAM gallery.

### XAI Insights

- Prediction
- Confidence Score
- Model identity
- Explanation text
- Temporal explanation
- Spatial explanation
- Grad-CAM and contact-sheet preview when those artifacts exist

If XAI was not produced, the screen says that XAI is unavailable. It does not invent images or scores.

### Tampering Map

- Video contact sheet
- The selected Grad-CAM frame
- Frame index
- Timestamp
- Face-crop or full-frame fallback information for that frame
- Selection among the available Grad-CAM frames

Image evidence keeps the existing image screens, including Model Confidence and the image heatmap / overlay viewer. Image routes stay on the image artifact endpoints. Video routes use the video contact sheet and `video_gradcam` frame artifacts.

Open the investigator UI at http://127.0.0.1:5000/evidex/ after the backend is running.

---

## Reports, audit timeline, and email

Generating a report creates one forensic PDF for the current analysis, then attempts to email that same PDF.

```
Generate Report
  → forensic PDF is created
  → PDF includes the current case and evidence
  → AI result and Confidence Score
  → video frame and fallback information, when the analysis is video
  → XAI artifacts when they exist
  → case-scoped audit timeline
  → the same PDF is emailed to the signed-in investigator
```

Image reports keep image Model Confidence, the image heatmap, and the image overlay.

Video reports keep the video Confidence Score, frame and fallback counts, the temporal contribution graph, and Grad-CAM when those artifacts were produced.

### Case-scoped audit timeline

The PDF timeline includes only audit events whose case is the case of the current analysis. It does not include other cases belonging to the same investigator, investigator-wide login history, or unrelated case events.

The investigator-facing audit API is separate. Admins can still review a broader audit log through that API. The report timeline is the case-scoped view.

### Automatic email

The recipient is the authenticated user’s registered `User.email`. There is no manual recipient field and no separate send button. The message is sent when the report is generated, not when the page is opened, refreshed, or when the PDF is downloaded.

Each generated report is emailed once. A report already marked sent is not sent again. The record stores email status and the sent timestamp.

If mail is disabled or SMTP delivery fails, the PDF is still created and can still be downloaded. The API reports that email delivery is unavailable. It does not describe a failed send as success.

SMTP settings come only from the server environment. Names only:

```
MAIL_ENABLED
MAIL_HOST
MAIL_PORT
MAIL_USERNAME
MAIL_PASSWORD
MAIL_FROM
MAIL_USE_TLS
```

Do not put real credentials in this file, in frontend JavaScript, or in Git. Copy `.env.example` to `.env` locally. `.env` is gitignored.

Report endpoints:

- `POST /api/analysis/{id}/report` — create the PDF and attempt delivery to the signed-in user
- `GET  /api/analysis/{id}/reports` — list reports for an analysis
- `GET  /api/reports/{id}` — report metadata
- `GET  /api/reports/{id}/download` — download the stored PDF

The stored path is resolved on the server and must stay inside the report directory. The client cannot supply an arbitrary PDF path.

---

## Password policy and rate limits

New passwords must have:

- at least 8 characters
- one uppercase letter
- one lowercase letter
- one number
- one special character

The registration form shows a live checklist and masks the password, with a show/hide control. The backend enforces the same rules and stores a Werkzeug password hash. Existing accounts are not disabled only because they were created before this policy. Login still checks the stored hash.

Rate limits use a 60-second window. They reduce credential stuffing and repeated heavy analysis or mail work. The current limiter is in memory and fits this local, single-process server. A multi-process deployment needs a shared limiter.

| Action | Default |
|---|---|
| Failed logins | 5 per 60 seconds, per client address |
| Registration | 10 per 60 seconds, per client address |
| Image analysis starts | 5 per 60 seconds, per user |
| Video analysis starts | 3 per 60 seconds, per user |
| Report email | 5 per 60 seconds, per user |

Successful logins do not consume the failed-login bucket. Viewing an existing analysis is not an analysis start. Report email uses its own bucket so mail limits and analysis limits do not spend each other.

---

## Security controls in this build

These controls are implemented in the application. This list is not a claim that every abuse case has been penetration-tested.

| Control | What the code does |
|---|---|
| Access | Flask-Login session required on protected APIs |
| Ownership | Case, evidence, analysis, and report actions check owner or admin |
| Passwords | Werkzeug hashing, plus strength checks on new passwords |
| Rate limits | Login, registration, image analysis, video analysis, and report email |
| Integrity | Server-side SHA-256 on upload; client-supplied hashes are not trusted |
| Files on disk | UUID names; downloads must resolve inside the storage root |
| Reports | Server-side path resolution; no client-chosen PDF path |
| Mail | SMTP credentials stay in server environment variables |
| Secrets in Git | `.env` is ignored |
| Local evidence | Uploads, generated reports, logs, and local databases are ignored |
| Errors | API errors use a safe JSON envelope; audit details scrub secret-like fields |
| Expensive image XAI | SHAP, fusion, and counterfactual run only when explicitly requested |

Sessions use HTTP-only, SameSite=Lax cookies. Further notes: [`docs/SECURITY.md`](docs/SECURITY.md).

---

## Running the application

Known local Windows workflow, from the repository root:

```powershell
cd C:\Users\sawar\OneDrive\Desktop\AI-Proof-Analyzer
.\.venv\Scripts\activate
python backend\run.py
```

The application listens on http://127.0.0.1:5000

- Health shell: http://127.0.0.1:5000/
- Health: http://127.0.0.1:5000/health
- EVIDEX: http://127.0.0.1:5000/evidex/
- API prefix: `/api`

Run one current Flask process for that port. An older process left on the same port can serve stale code while a newer terminal looks idle.

First-time setup, if `.venv` does not exist yet:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
copy .env.example .env
python backend\run.py
```

Hardware expectation: Windows 11 or Linux, 8 GB RAM, CPU-first. A dedicated GPU is not required. Prefer `num_workers=0` for DataLoader defaults.

### Frontends

| Directory | Purpose | Served at |
|---|---|---|
| `DIGITALEVIDENCE_FIXED/` | EVIDEX investigator UI | `/evidex/` |
| `frontend/` | Minimal Jinja health shell from Phase 1 | `/` and `/static` |

EVIDEX is plain HTML, CSS, and JavaScript. There is no npm build. Flask serves it from the same origin so the session cookie stays first-party.

### Docker

CPU-first container with volumes for the database, uploads, reports, logs, and investigation artifacts:

```bash
docker-compose up --build -d
docker-compose ps
curl http://127.0.0.1:5000/health
docker-compose down
```

Deployment notes: [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md). The local Windows command above is the documented developer workflow. Do not treat an old alternate port as the application port.

---

## What belongs in Git

The repository holds application source, tests, documentation, and the two production video checkpoints:

```
artifacts/checkpoints/video/visual_model.pt
artifacts/checkpoints/video/lstm.pt
```

Local and generated material stays out of Git, including:

- `.env`
- `.venv/`
- datasets and local image collections
- uploads
- generated reports
- logs
- local databases
- temporary files
- Postman local workspace state
- unrelated checkpoints, including `artifacts/checkpoints/processed_final/`

`.env.example` documents variable names only. Never commit a live SMTP password or other credential.

---

## Testing

Completed product checks cover:

- EVIDEX frontend UI, including the three video screens and the image heatmap path
- authentication and password policy
- rate limits
- image-analysis regression
- the video pipeline, including face-crop and full-frame fallback
- video XAI
- video and image report generation
- automatic report email, with SMTP mocked in tests
- the case-scoped audit timeline
- end-to-end video UI rendering

Examples:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_auth_api.py tests/test_password_security.py tests/test_video_e2e.py tests/test_video_xai.py tests/test_video_report.py tests/test_report_email.py tests/test_audit.py -q
node --test tests/test_video_result_ui.js tests/test_password_policy_ui.js tests/test_report_email_ui.js
```

Some older dataset, training, evaluation, and SHAP tests need a local processed dataset (`MAYA_PROCESSED_DATASET_DIR`), extra checkpoints, or optional packages such as `shap`. Those tests are still in the tree. A failure there does not mean the product video pipeline is unconfigured. Do not treat the full optional suite as green unless that local setup is actually present.

Image end-to-end coverage remains `tests/test_e2e_product.py`.

---

## Image dataset and image model

The image corpus workflow is unchanged. `scripts/build_final_dataset.py` builds the final image dataset from IMD2020:

- `*_orig.jpg` images are REAL; other non-mask images are FAKE
- cases stay isolated across train, validation, and test
- SHA-256 deduplication
- masks excluded
- standardized 224×224 RGB JPEG
- a separate external evaluation set from unseen cases
- leakage and integrity audits
- source datasets are not modified or deleted

| Split | Cases | REAL | FAKE | Total |
|---|---:|---:|---:|---:|
| Train | 246 | 245 | 1,738 | 1,983 |
| Validation | 52 | 52 | 78 | 130 |
| Test | 54 | 53 | 130 | 183 |
| **Primary total** | **352** | **350** | **1,946** | **2,296** |

| Set | Cases | REAL | FAKE | Total |
|---|---:|---:|---:|---:|
| External | 62 | 62 | 62 | 124 |

The final image-dataset audit reported zero violations for SHA cross-partition overlap, case overlap, duplicate-component overlap, external/primary overlap, mask contamination, missing outputs, and manifest/count mismatch.

Details: [`docs/DATASET.md`](docs/DATASET.md) · [`docs/DATASET_VERSIONING.md`](docs/DATASET_VERSIONING.md)

Image training uses EfficientNet-B0, ImageNet initialization, fine-tuning of the final two feature blocks, differential learning rates, and case-balanced sampling:

```powershell
$env:MAYA_PROCESSED_DATASET_DIR = "$PWD\dataset\final"
python scripts/train.py --profile production --device cpu --notes "Final IMD2020 training with case-balanced sampling and final-two-block EfficientNet fine-tuning"
```

Image evaluation and prediction entry points remain `scripts/evaluate.py`, `scripts/evaluate_final_independent.py`, and `scripts/predict.py`. Those commands belong to the image model. They do not replace the frozen video checkpoints.

---

## Image explainability (Phase 4)

Plugin-based image XAI lives under `ai/explainability/`:

| Sprint | Role |
|---|---|
| 4.1 | Grad-CAM foundation and explanation artifacts |
| 4.2 | Grad-CAM++, LayerCAM, ScoreCAM, EigenCAM |
| 4.3 | Focus, localization, quality, and trust analytics |
| 4.4 | Explainer benchmark and ranking |
| 4.5 | SHAP, faithfulness, counterfactual, fusion, audit |

```python
from ai.explainability import ExplainabilityEngine, ExplainabilityConfig

ExplainabilityEngine(
    ExplainabilityConfig(explainer_name="gradcam", device_preference="cpu")
).explain(r"path\to\image.jpg")
```

Artifacts: `artifacts/phase4/sprint1/` through `artifacts/phase4/sprint5/`. Notes: [`docs/PHASE4_SPRINT1.md`](docs/PHASE4_SPRINT1.md) through [`docs/PHASE4_SPRINT5.md`](docs/PHASE4_SPRINT5.md).

---

## Product API

Protected routes use the Flask-Login API decorator. Full reference: [`docs/API.md`](docs/API.md).

### Auth

- `POST /api/auth/register`
- `POST /api/auth/login`
- `POST /api/auth/logout`
- `GET  /api/auth/me`

### Cases

- `POST /api/cases`
- `GET  /api/cases`
- `GET  /api/cases/{id}`
- `PATCH /api/cases/{id}`
- `POST /api/cases/{id}/close`

### Evidence

- `POST /api/evidence/cases/{id}` — multipart upload; SHA-256 is computed on the server
- `GET  /api/evidence/cases/{id}`
- `GET  /api/evidence/{id}`
- `POST /api/evidence/{id}/verify-integrity`

### Analysis

- `POST /api/evidence/{id}/analyze` — image EfficientNet-B0, or the 16-frame video pipeline when the evidence is video
- `GET  /api/analysis/{id}`
- `GET  /api/investigations/{id}` — alias of the analysis read

Image requests may opt into extra explainers:

```json
{
  "generate_explanation": true,
  "explainer": "gradcam",
  "verify_before_analyze": true,
  "advanced_xai": {
    "shap": true,
    "faithfulness": true,
    "counterfactual": true,
    "fusion": true,
    "trust": true
  }
}
```

### Audit API

- `GET /api/audit` — an admin sees the broader log; an investigator sees their own events

That API is not the PDF timeline. Report PDFs are filtered to the current analysis case only.

Product notes: [`docs/PHASE3_PRODUCT.md`](docs/PHASE3_PRODUCT.md) · [`docs/PHASE3_PRODUCT_ARCHITECTURE.md`](docs/PHASE3_PRODUCT_ARCHITECTURE.md)

---

## Investigation workflow

1. Create a case.
2. Upload evidence.
3. Verify integrity.
4. Analyze. Image evidence uses the image model. Video evidence uses the 16-frame LSTM pipeline.
5. Review Analysis, XAI Insights, and Tampering Map in EVIDEX. Image evidence keeps the image heatmap and overlay.
6. Generate the PDF. The same file is emailed to the signed-in investigator when SMTP is configured.

Each investigation can store artifacts under `artifacts/investigations/`. Evidence and reports are SHA-256 sealed and audit-logged.

Guide: [`docs/INVESTIGATION_WORKFLOW.md`](docs/INVESTIGATION_WORKFLOW.md) · [`docs/VIVA_GUIDE.md`](docs/VIVA_GUIDE.md)

---

## Repository layout

```
AI-Proof-Analyzer/
├── ai/
│   ├── datasets/           # Image corpus pipeline and DataLoaders
│   ├── models/             # Image EfficientNet-B0
│   ├── training/           # Image training CLI
│   ├── evaluation/         # Image metrics and offline eval
│   ├── inference/          # Image investigation prediction
│   ├── ffpp_video/         # Production 16-frame visual encoder, LSTM, face fallback, Grad-CAM
│   ├── video/              # Video frame aggregation and temporal helpers
│   ├── benchmark/
│   ├── engine/
│   └── explainability/     # Image XAI
├── backend/app/
│   ├── api/
│   ├── services/           # Analysis, video, reports, email
│   ├── models/
│   ├── security/           # Passwords and rate limits
│   └── config/
├── DIGITALEVIDENCE_FIXED/  # EVIDEX UI
├── frontend/               # Phase 1 health shell
├── artifacts/
│   ├── checkpoints/video/  # visual_model.pt and lstm.pt
│   ├── phase3/             # Image train / eval / infer outputs
│   ├── phase4/             # Image explainability artifacts
│   └── investigations/     # Per-investigation runtime bundles (contents gitignored)
├── dataset/                # Scaffolding tracked; image blobs local
├── docs/
├── tests/
├── scripts/
├── uploads/                # Local evidence (gitignored)
├── reports/                # Generated PDFs (gitignored)
└── logs/                   # Local logs (gitignored)
```

Database entities in `backend/app/models/entities.py`:

- **User** — email, username, role, password hash
- **Case** — `CASE-{year}-{seq}`, status, owner
- **Evidence** — UUID filename, server-computed SHA-256, case
- **AnalysisRun** — prediction, confidence, video analysis payload, XAI paths
- **AuditLog** — append-only events, including case id
- **InvestigationReport** — `RPT-{year}-{seq}`, SHA-256, storage path, email status, sent timestamp

Active image-dataset pointer: `dataset/versions/CURRENT`.

Architecture rule: code under `ai/` does not import Flask. The backend calls into `ai/` through service and integration layers.

---

## Known limitations

- The published binary video metrics come from a limited FF++ test split.
- That split does not represent all real-world videos.
- The production video classifier is binary REAL / FAKE.
- Manipulation-type classification is not part of the current production pipeline.
- Face detection can miss a face.
- Full-frame fallback keeps the frame in the sequence. It is not a fake detector.
- Grad-CAM and temporal XAI explain the model. They are not proof of manipulation.
- Confidence Score is the predicted-class probability, not legal certainty.
- In-memory rate limiting matches the current single-process local server. Multi-process deployment needs shared limit state.
- Report email is sent only when SMTP is configured and the server accepts the message. The PDF still exists when delivery fails.

---

## Documentation index

Start at [`docs/00_PHASE0_INDEX.md`](docs/00_PHASE0_INDEX.md).

Phase 3: [`docs/PHASE3_SPRINT1.md`](docs/PHASE3_SPRINT1.md) through [`docs/PHASE3_SPRINT5.md`](docs/PHASE3_SPRINT5.md)

Phase 4: [`docs/PHASE4_SPRINT1.md`](docs/PHASE4_SPRINT1.md) through [`docs/PHASE4_SPRINT5.md`](docs/PHASE4_SPRINT5.md)

Also: [`docs/API.md`](docs/API.md) · [`docs/SECURITY.md`](docs/SECURITY.md) · [`docs/DATABASE.md`](docs/DATABASE.md) · [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) · [`docs/TESTING.md`](docs/TESTING.md) · [`docs/XAI_ARCHITECTURE.md`](docs/XAI_ARCHITECTURE.md)

---

## License / use

Intended for academic and authorized investigative training. This is not a consumer public scanner, and a model output is not a legal determination.
