# MAYA — Media Authenticity Analyzer

AI-powered **digital evidence investigation platform** for authenticity assessment, explainable analysis, and professional investigation workflows.

> Deepfake detection is one component—not the whole product.

## Current status

| Phase | Status |
|-------|--------|
| Phase 0 — Design | Complete (`docs/`) |
| Phase 1 — Foundation | Complete |
| Phase 2 — Evidence data engineering | Complete |
| Phase 2.5 — Dataset pipeline review & optimization | Complete |
| Phase 3.1 — AI model architecture | Complete |
| Phase 3.2 — AI training engine | Complete |
| Phase 3.3 — AI validation & reporting | Complete |
| Phase 3.4 — Investigation inference | Complete |
| Phase 3.5 — AI performance benchmarks | Complete |
| Phase 4.1 — Explainability (Grad-CAM foundation) | Complete |
| Phase 4.2 — Multi-Explainer Framework | Complete |
| Phase 4.3 — Explanation Analytics & Trust | Complete |
| Phase 4.4 — Explainability Validation & Benchmark | Complete |
| Phase 4.5 — Advanced Explainability & Trust Layer | Complete |
| Phase 3 Product — Auth / Cases / Evidence / APIs | Complete |
| Phase 5 — Reports / Hardening / Docker / E2E Tests | Complete |
| Phase 5 — Face reference verification (backend) | Complete |

Roadmap: [`docs/ROADMAP.md`](docs/ROADMAP.md)

---

## Hardware requirements

- Windows 11 / Linux (Docker)
- **8 GB RAM**
- CPU-first (no dedicated GPU required)
- Prefer `num_workers=0` DataLoader defaults

---

## Quick start (Local development)

```bash
# From repository root
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Install CPU-optimized PyTorch (smaller, no NVIDIA needed)
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

copy .env.example .env

# Initialize database + run
python backend/run.py
```

- Home (health shell): http://127.0.0.1:5000/
- Health: http://127.0.0.1:5000/health
- **EVIDEX investigator UI: http://127.0.0.1:5000/evidex/**
- API prefix: `/api/*` (JSON, Flask-Login session cookies)

### Frontends

Two separate frontends live in this repository:

| Directory | Purpose | Served at |
|---|---|---|
| `DIGITALEVIDENCE_FIXED/` | **EVIDEX** — the investigator/forensic UI, integrated with the MAYA API | `/evidex/` |
| `frontend/` | Minimal Jinja health/test shell from Phase 1 | `/` and `/static` |

EVIDEX is plain HTML/CSS/JS with no build step — there is nothing to `npm
install`. It is served from the Flask origin so the Flask-Login session cookie
stays first-party and no CORS configuration is needed. Just start the backend
and open `/evidex/`.

To serve it from a different origin instead, set `window.EVIDEX_API_BASE`
before `script.js` loads and add the matching CORS configuration; the API base
URL is centralised in `DIGITALEVIDENCE_FIXED/api.js` and is not hardcoded
anywhere else.

The investigator pages (dashboard, cases, upload, analysis, XAI, heat map,
chain of custody, evidence readiness, reports, profile) run against the real
backend. The admin console and the public verification preview are still backed
by seeded browser-local demo data and are labelled as such in the UI.

---

## Docker deployment (Phase 5)

CPU-first container with persistent volumes (database, uploads, reports, logs, investigation artifacts).

```bash
# Build + start
docker-compose up --build -d

# Check health
docker-compose ps
curl http://127.0.0.1:5000/health

# Stop + keep volumes
docker-compose down
```

Volumes mounted:
- `maya-db` → SQLite DB
- `maya-uploads` → Evidence files
- `maya-reports` → Generated PDF reports
- `maya-logs` → Server logs
- `maya-investigations` → AI/XAI investigation artifacts

Docs: [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)

---

## Dataset workflow

```bash
# 1) Place Kaggle REAL/FAKE images in dataset/raw/
#    or materialize from catalogue CSV, or:
#    $env:MAYA_RAW_DATASET_DIR = "D:\path\to\extract"

# 2) Build balanced 224×224 corpus + reports
python scripts/run_dataset_pipeline.py

# 3) (Optional) Seal / re-seal version metadata for an existing processed set
python scripts/seal_dataset_version.py

# 4) Verify Dataset / DataLoader / versioning
python -m pytest tests/test_dataset.py -q
```

Details: [`docs/DATASET.md`](docs/DATASET.md) · [`docs/DATASET_VERSIONING.md`](docs/DATASET_VERSIONING.md)

---

## AI Pipeline

### Training (Phase 3.2)

```bash
python scripts/train.py --profile debug
python scripts/train.py --profile development
python scripts/train.py --profile production

python -m pytest tests/test_training.py -q
```

Logs: `logs/training.log` · Artefacts: `artifacts/phase3/sprint2/`

### Evaluation (Phase 3.3)

```bash
python scripts/evaluate.py --threshold 0.5
python -m pytest tests/test_evaluation.py -q
```

Artefacts: `artifacts/phase3/sprint3/`

### Inference (Phase 3.4)

```bash
python scripts/predict.py path\to\image.jpg
python scripts/predict.py --folder path\to\images
python -m pytest tests/test_inference.py -q
```

Artefacts: `artifacts/phase3/sprint4/`

### Inference benchmarks (Phase 3.5)

```bash
python scripts/benchmark.py
python scripts/benchmark.py --runs 20
python -m pytest tests/test_benchmark.py -q
```

Artefacts: `artifacts/phase3/benchmark/`

---

## Explainability (Phase 4)

Plugin-based XAI stack under `ai/explainability/`:

| Sprint | What it does |
|--------|----------------|
| 4.1 | Grad-CAM foundation + explanation artefacts |
| 4.2 | Grad-CAM++, LayerCAM, ScoreCAM, EigenCAM + comparison |
| 4.3 | Focus / localization / quality / trust analytics |
| 4.4 | Explainer benchmark, ranking, recommendations |
| 4.5 | SHAP, faithfulness, counterfactual, fusion, audit |

```python
# Single explanation (Sprint 4.1+)
from ai.explainability import ExplainabilityEngine, ExplainabilityConfig

result = ExplainabilityEngine(
    ExplainabilityConfig(explainer_name="gradcam", device_preference="cpu")
).explain(r"path\to\image.jpg")

# Multi-explainer comparison (Sprint 4.2)
from ai.explainability import ExplainabilityEngine, ExplainabilityConfig

ExplainabilityEngine(ExplainabilityConfig(device_preference="cpu")).compare(
    r"path\to\image.jpg"
)

# Analytics on heatmaps (Sprint 4.3)
from ai.explainability.analytics import ExplanationAnalyticsEngine, AnalyticsConfig

ExplanationAnalyticsEngine(AnalyticsConfig()).analyze_from_heatmap_images(
    {"gradcam": r"artifacts\phase4\sprint2\gradcam_heatmap.png"},
    prediction="FAKE",
    model_confidence=80.0,
)

# Rank all registered explainers (Sprint 4.4)
from ai.explainability.benchmark import ExplainabilityBenchmarkSuite

ExplainabilityBenchmarkSuite().run(r"path\to\image.jpg")

# Advanced XAI: SHAP + faithfulness + counterfactual + trust (Sprint 4.5)
from ai.explainability import AdvancedExplainabilityEngine, AdvancedXAIConfig

AdvancedExplainabilityEngine(AdvancedXAIConfig(device_preference="cpu")).analyze(
    r"path\to\image.jpg"
)
```

```bash
python -m pytest tests/test_gradcam.py tests/test_multi_explainer.py `
  tests/test_explanation_analytics.py tests/test_explainability_benchmark.py `
  tests/test_shap.py tests/test_faithfulness.py tests/test_counterfactual.py `
  tests/test_fusion.py tests/test_trust.py -q
```

Artefacts: `artifacts/phase4/sprint{1..5}/`
Docs: [`PHASE4_SPRINT1.md`](docs/PHASE4_SPRINT1.md) · [`SPRINT2`](docs/PHASE4_SPRINT2.md) · [`SPRINT3`](docs/PHASE4_SPRINT3.md) · [`SPRINT4`](docs/PHASE4_SPRINT4.md) · [`SPRINT5`](docs/PHASE4_SPRINT5.md)

---

## Product APIs (Phase 3 + Phase 5)

Flask-Login sessions + SQLAlchemy models under `backend/app/`. All protected routes use `@login_required_api` decorator.

```bash
python backend/run.py
```

### Full API reference: [`docs/API.md`](docs/API.md)

### Auth
- `POST /api/auth/register` — Register user (if `ALLOW_PUBLIC_REGISTRATION=true`)
- `POST /api/auth/login` — Start session
- `POST /api/auth/logout` — End session
- `GET  /api/auth/me` — Current user profile

### Cases
- `POST /api/cases` — Create investigation case
- `GET  /api/cases` — List own cases (admin sees all)
- `GET  /api/cases/{id}` — Case details (ownership enforced)
- `PATCH /api/cases/{id}` — Update case metadata
- `POST /api/cases/{id}/close` — Close case

### Evidence
- `POST /api/evidence/cases/{id}` — Upload evidence (multipart/form-data) — **SHA-256 computed server-side; client hashes NEVER trusted
- `GET  /api/evidence/cases/{id}` — List evidence in case
- `GET  /api/evidence/{id}` — Evidence metadata
- `POST /api/evidence/{id}/verify-integrity` — Recompute SHA-256 on-disk; compare with stored hash

### Analysis / Investigation (real AI inference + XAI)
- `POST /api/evidence/{id}/analyze` — Run EfficientNet-B0 inference + optional XAI
- `GET  /api/analysis/{id}` — Retrieve analysis results
- `GET  /api/investigations/{id}` — Alias for above

Analysis request body (opt-in expensive methods:
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

### PDF Reports (Phase 5)
- `POST /api/analysis/{id}/report` — Generate signed, hashed investigation PDF
- `GET  /api/analysis/{id}/reports` — List reports for an analysis
- `GET  /api/reports/{id}` — Report metadata
- `GET  /api/reports/{id}/download` — Binary PDF download (path-traversal-safe)

Report includes: case & evidence metadata, authenticity assessment, SHA-256 integrity, XAI visualizations, advanced XAI artifact refs, audit timeline, investigator notes, and standard disclaimer. Source: [`report_service.py`](backend/app/services/report_service.py)

### Audit Trail
- `GET /api/audit` — ADMIN: all events; regular user: own events only

Event types: `USER_REGISTERED`, `USER_LOGIN`, `USER_LOGOUT`, `CASE_CREATED`, `CASE_UPDATED`, `CASE_CLOSED`, `EVIDENCE_UPLOADED`, `EVIDENCE_ACCESSED`, `EVIDENCE_VERIFIED`, `ANALYSIS_STARTED`, `ANALYSIS_COMPLETED`, `ANALYSIS_FAILED`, `XAI_GENERATED`, `REPORT_GENERATED`

### Product API tests:
```bash
python -m pytest tests/test_auth_api.py tests/test_cases_api.py `
  tests/test_evidence_api.py tests/test_analysis_api.py -q
```

Product docs: [`PHASE3_PRODUCT.md`](docs/PHASE3_PRODUCT.md) · [`PHASE3_PRODUCT_ARCHITECTURE.md`](docs/PHASE3_PRODUCT_ARCHITECTURE.md)

---

## End-to-End tests (Phase 5)

Genuine E2E flow with real AI inference (non-mocked):

```bash
python -m pytest tests/test_e2e_product.py -q -v
```

Covered flows:
1. Register → Login → Create Case → Upload Evidence → **Verify SHA-256 integrity → **Run real EfficientNet-B0 inference → **Persist to DB → **Audit trail → **Artifact directory creation
2. Cross-user authorization enforcement (403 at every layer)
3. Tampered evidence integrity detection (MODIFIED)

Diagnostic script:
```bash
python scripts/_diag_product_ai.py
```

---

## Security (Phase 5 hardening)

| Control | Implementation |
|---|---|
| Passwords | Werkzeug argon2-family hashing |
| Sessions | HTTP-only, SameSite=Lax cookies |
| Authorization | Owner OR ADMIN at every service call (cases/evidence/analysis/report) |
| Integrity | Server SHA-256 on upload + re-verify API; client hashes ignored |
| Filenames | UUID-based; never client names on disk |
| Path traversal | `.resolve()` + `startswith(root)` on all downloads |
| Error handling | Safe JSON envelope; stack traces never leak to client |
| Audit logs | Append-only; secrets (password/token/secret/auth) scrubbed before insert |
| Uploads | MIME + extension + size double-checked |
| DoS gates | SHAP/fusion/counterfactual only on explicit `advanced_xai` opt-in |

Full details: [`docs/SECURITY.md`](docs/SECURITY.md)

---

## Processing pipeline

```
raw/ → inventory → integrity → statistics/plots
     → seeded sample → preprocess (224 RGB)
     → validate → seal versions/v1 + dataset_metadata.json
```

PyTorch access:

```python
from ai.datasets import create_dataloader, build_split_dataset
from ai.datasets.dataset_config import SplitName

loader = create_dataloader(SplitName.TRAIN, batch_size=16, transform_name="train")
```

---

## Folder structure

```
MAYA/
├── ai/
│   ├── datasets/          # Corpus pipeline + DataLoaders + versioning
│   ├── models/            # EfficientNet-B0 + model factory
│   ├── training/          # Training CLI / callbacks / logging
│   ├── evaluation/        # Metrics + offline eval
│   ├── inference/         # Investigation prediction pipeline
│   ├── benchmark/     # Inference performance suite
│   ├── engine/            # Trainer + checkpoint + experiment history
│   └── explainability/    # Phase 4 XAI (explainers, analytics, benchmark, SHAP, fusion, trust)
├── backend/               # Flask application (product APIs)
│   ├── app/
│   │   ├── api/            # JSON route blueprints (auth/cases/evidence/analysis/audit)
│   │   ├── services/       # Business logic + ownership checks
│   │   ├── models/         # SQLAlchemy entities + enums
│   │   ├── audit/        # Append-only audit service
│   │   ├── integrations/ # AI bridge (backend→ai/)
│   │   ├── storage/      # Evidence file storage
│   │   ├── security/    # Password + session helpers
│   │   ├── config/      # Config classes
│   │   ├── database/  # DB init
│   │   └── exceptions.py  # Safe error model
├── frontend/              # Templates & static assets (simple web shell)
├── dataset/
│   ├── raw/               # Immutable source
│   ├── versions/       # Sealed metadata + CURRENT pointer
│   └── reports/          # Dataset pipeline reports
├── artifacts/
│   ├── phase3/            # Train / eval / infer / bench outputs
│   ├── phase4/            # Explainability sprint artefacts
│   └── investigations/  # Per-case AI/XAI investigation runs (INV-{ID}/)
├── docs/
├── tests/
├── scripts/
├── uploads/             # Evidence uploads (gitignored, Docker volume)
├── reports/             # Generated PDFs (gitignored, Docker volume)
├── logs/                # Server logs (gitignored, Docker volume)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Database entities (SQLAlchemy)

- **User** — id, email, username, role (INVESTIGATOR/ADMIN), password_hash
- **Case** — CASE-{year}-{seq} case_number, status (OPEN/IN_PROGRESS/CLOSED/ARCHIVED), priority, owner FK
- **Evidence** — stored_filename (UUID), storage_path, **sha256_hash (server-computed), file_size, mime_type, status, case FK
- **AnalysisRun** — INV-{year}-{seq} investigation_id, prediction, confidence, {trust/quality scores, heatmap/overlay paths, advanced_xai JSON, evidence FK
- **AuditLog** — Append-only event log (timestamp, event_type, FK refs to case/evidence/analysis/user, scrubbed details
- **InvestigationReport** — RPT-{year}-{seq} report_number, **sha256, storage_path, PDF metadata

Schema: [`entities.py`](backend/app/models/entities.py)

---

## Dataset versioning

Active version pointer: `dataset/versions/CURRENT`
Sealed metadata: `dataset/versions/v1/dataset_metadata.json`

Future corpora (**FaceForensics++**, **Celeb-DF**) plug in via config / `MAYA_RAW_DATASET_DIR` without changing pipeline architecture.

---

## Investigation Workflow

1. **Create Case → **Upload Evidence** → **Verify Integrity → **Analyze** (AI + XAI) → **Generate PDF Report**

Each investigation gets a unique `INV-{ID}` directory under `artifacts/investigations/` containing: prediction JSON, heatmaps, overlays, SHAP visualizations, and advanced XAI outputs. All evidence and reports are SHA-256 sealed and audit-logged.

Guide: [`INVESTIGATION_WORKFLOW.md`](docs/INVESTIGATION_WORKFLOW.md) · [`VIVA_GUIDE.md`](docs/VIVA_GUIDE.md)

---

## Documentation

Start here: [`docs/00_PHASE0_INDEX.md`](docs/00_PHASE0_INDEX.md)

Phase 3 sprint notes: [`PHASE3_SPRINT1`](docs/PHASE3_SPRINT1.md)–[`SPRINT5`](docs/PHASE3_SPRINT5.md)
Phase 4 sprint notes: [`PHASE4_SPRINT1`](docs/PHASE4_SPRINT1.md)–[`SPRINT5`](docs/PHASE4_SPRINT5.md)

Other: [`API.md`](docs/API.md) · [`SECURITY.md`](docs/SECURITY.md) · [`DATABASE.md`](docs/DATABASE.md) · [`DEPLOYMENT.md`](docs/DEPLOYMENT.md) · [`TESTING.md`](docs/TESTING.md) · [`XAI_ARCHITECTURE.md)](docs/XAI_ARCHITECTURE.md)

---

## Architecture

**Presentation → Application → Business Logic → AI Analysis → Storage**

AI code under `ai/` must not import Flask. Explainability is independent of training/eval/benchmark packages and is requested by higher layers when needed.

---

## License / use

Intended for academic and authorized investigative training contexts. Not a consumer public scanner.
