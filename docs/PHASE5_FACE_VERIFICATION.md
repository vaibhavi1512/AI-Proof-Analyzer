# MAYA — Phase 5: Face Reference Verification

Backend-only investigator workflow: upload a **reference face**, compare it to a face detected in case **evidence**, persist a forensic result, write investigation artefacts, emit audit events, and optionally include the result in the existing PDF report.

## Pipeline

Reference image → detect → quality → embed  
Evidence image → detect → quality → embed  
→ cosine similarity + Euclidean distance → threshold decision → `MATCH` / `NO_MATCH` / `INCONCLUSIVE`

Verification is **not** forced to a binary outcome. Zero faces, multiple faces, low-quality crops, corrupted/unsupported images, and mid-band similarity scores yield `INCONCLUSIVE` (or `FAILED` only when detection/embedding raises).

## Library choice

**Default engine:** OpenCV Haar cascade (`cv2.data.haarcascades`) + an 8×8 RGB-grid embedding. Uses **opencv-python-headless** already in MAYA (Phase 4.5). No extra package, no model download, Windows-safe, compatible with the current PyTorch install.

**Optional engine:** `facenet-pytorch` (MTCNN + InceptionResnetV1 / VGGFace2) when you explicitly want FaceNet embeddings:

```powershell
.\.venv\Scripts\python.exe -m pip install facenet-pytorch --no-deps
```

`--no-deps` is required: facenet-pytorch 2.6.0 otherwise pins `torch<2.3` and `numpy<2`, which would break MAYA’s existing stack. Then set `FACE_VERIFICATION_ENGINE=facenet_pytorch`.

Tests mock the engine and do not download FaceNet weights.

## Configuration

Set in `.env` or Flask config (see `.env.example`):

| Variable | Default | Meaning |
|----------|---------|---------|
| `FACE_VERIFICATION_ENGINE` | `facenet_pytorch` | `facenet_pytorch` or `opencv` |
| `FACE_VERIFICATION_MODEL` | `inception_resnet_v1` | Recorded model name |
| `FACE_VERIFICATION_MODEL_VERSION` | `vggface2` | Recorded version |
| `FACE_VERIFICATION_PRETRAINED` | `vggface2` | FaceNet pretrained set (`vggface2` / `casia-webface`) |
| `FACE_VERIFICATION_DEVICE` | `cpu` | Torch device preference |
| `FACE_VERIFICATION_MATCH_THRESHOLD` | `0.70` | Cosine similarity ≥ this → `MATCH` |
| `FACE_VERIFICATION_NO_MATCH_THRESHOLD` | `0.50` | Cosine similarity < this → `NO_MATCH` |
| `FACE_VERIFICATION_MIN_FACE_SIZE` | `40` | Minimum crop edge (pixels) |
| `FACE_VERIFICATION_MIN_SHARPNESS` | `25.0` | Laplacian variance floor |
| `FACE_VERIFICATION_CACHE_DIR` | unset | Optional weights cache directory |

Scores between the two thresholds are `INCONCLUSIVE`. A request may override thresholds via multipart form fields `threshold` and `no_match_threshold` (both in `[0, 1]`).

## API

Session cookie auth (same as other product APIs). Case/evidence **ownership** is enforced.

### `POST /api/evidence/{evidence_id}/face-verification`

Multipart:

- `file` — reference image (same allowlist as evidence: jpg/jpeg/png/bmp/webp)
- `threshold` (optional)
- `no_match_threshold` (optional)
- `investigation_id` (optional; must belong to this evidence or case)

**201** `{ "ok": true, "data": { ... } }`

### `GET /api/face-verifications/{verification_id}`

**200** same `data` object. Cross-user access → 403.

Public `data` fields include decision, scores, thresholds, model identifiers, face counts, relative `artifact_dir`, and `reason_code`. Absolute OS paths of temp uploads are not returned.

## Artefacts

```
artifacts/investigations/<INVESTIGATION_ID>/face_verification/<run-token>/
  reference_face.png
  evidence_face.png
  comparison_result.json
  verification_report.json
```

If the evidence already has an analysis `investigation_id`, that ID is reused. Otherwise a new `INV-YYYY-NNNNNN` is allocated under `artifacts/investigations/investigation_id_state.json`.

## Audit

`FACE_VERIFICATION_STARTED`, `FACE_VERIFICATION_COMPLETED`, `FACE_VERIFICATION_FAILED` via the existing `audit_logs` table (`verification_id` in `details`).

## Reports

Existing `POST /api/analysis/{id}/report` is unchanged. When a face-verification row exists for the investigation or evidence, the PDF adds a **Face Reference Verification** section. Analyses without face verification generate reports as before.

## Database

New table `face_verifications` created with the existing `db.create_all()` + SQLite column-ensure path. No drop/recreate of the production database.
