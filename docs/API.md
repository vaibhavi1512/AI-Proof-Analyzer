# MAYA API Documentation (Backend JSON API — JSON REST API served under the `/api` prefix. Uses Flask-Login session cookies; clients POST credentials once per session.

All responses use the envelope:

```json
{"ok": true, "data": ..., "message": "optional"}
{"ok": false, "error": "error_code", "message": "Human readable"}
```

## Authentication header: none (Flask sessions cookie). Cookie is set after `Set-Cookie` header returned by `/api/auth/login`.

## 1. Auth (`/api/auth`)

### `POST /api/auth/register`

- **Auth:** Public (unless `ALLOW_PUBLIC_REGISTRATION=false`)
- **Body (JSON):** `{email, username, password, full_name?}`
- **Success (201):** User object
- **Errors:** 400 validation, 409 conflict

### `POST /api/auth/login`

- **Auth:** Public
- **Body (JSON):** `{login: email_or_username, password}`
- **Success (200):** User object + session cookie
- **Errors:** 401 authentication_error

### `POST /api/auth/logout`

- **Auth:** Required
- **Success (200):** `{ok:true, message:"Logged out"}`

### `GET /api/auth/me`

- **Auth:** Required
- **Success (200):** Current user object

User object shape:

```json
{
  "id": 1,
  "email": "analyst@maya.test",
  "username": "analyst",
  "full_name": "Test User",
  "role": "INVESTIGATOR | ADMIN",
  "is_active": true,
  "created_at": "2026-08-30T12:00:00",
  "last_login_at": "..."
}
```

## 2. Cases (`/api/cases`)

### `POST /api/cases`

- **Auth:** Required
- **Body:** `{title, description?, priority?: LOW|MEDIUM|HIGH}`
- **201:** Case object
- Ownership: created_by_user_id = current user

### `GET /api/cases`

- **Auth:** Required
- **Returns:** Array of case objects; ADMIN sees all; INVESTIGATOR only own cases.

### `GET /api/cases/{case_id}`

- **Auth:** Required (ownership check)
- **Returns:** Case object or 403/404.

### `PATCH /api/cases/{case_id}`

- **Auth:** Required (owner or admin)
- **Body:** `{title?, description?, status?, priority?}`
- Cannot modify ARCHIVED cases.

### `POST /api/cases/{case_id}/close`

- **Auth:** Required (owner or admin)
- Sets status=CLOSED + closed_at timestamp.

### `DELETE /api/cases/{case_id}`

- **Auth:** Required (owner or admin)
- Deletes the case together with its evidence, analysis runs, face verifications
  and investigation reports.
- Audit rows are **retained** (the custody trail outlives the record), and stored
  evidence files on disk are **not** removed.
- **200:** `{case_id, case_number, deleted_evidence, deleted_analyses, deleted_face_verifications, deleted_reports}`
- **403:** `authorization_error` when the caller neither owns the case nor is an admin
- **404:** `not_found` for an unknown or already-deleted case

Case object:

```json
{
  "case_id": 1,
  "case_number": "CASE-2026-000001",
  "title": "..",
  "description": "..",
  "status": "OPEN | IN_PROGRESS | CLOSED | ARCHIVED",
  "priority": "MEDIUM",
  "created_by": 1,
  "created_at": "...",
  "updated_at": "...",
  "closed_at": null
}
```

## 3. Evidence (`/api/evidence`)

### `POST /api/evidence/cases/{case_id}`

- **Auth:** Required (case owner)
- **Content-Type:** `multipart/form-data`
- **Fields:**
  - `file`: image file (`.png/jpg/jpeg/bmp/webp; max 16 MB)
  - `notes` (optional string): human notes
- **201:** Evidence metadata + SHA-256 computed server-side
- Client-supplied hashes are NEVER trusted.

### `GET /api/evidence/cases/{case_id}`

- **Auth:** Required (ownership)
- List all evidence items for a case.

### `GET /api/evidence/{evidence_id}`

- **Auth:** Required (ownership check)
- Evidence metadata.

### `POST /api/evidence/{evidence_id}/verify-integrity`

- **Auth:** Required (ownership check)
- Recomputes SHA-256 of the on-disk file; compares to `evidence.sha256_hash`.
- Returns:

```json
{
  "evidence_id": 5,
  "stored_sha256": "original-hash...",
  "current_sha256": "current-hash-or-empty",
  "integrity_status": "VALID | MODIFIED | MISSING | ERROR"
}
```

Evidence object:

```json
{
  "evidence_id": 1,
  "case_id": 1,
  "original_filename": "evidence.png",
  "stored_filename": "uuid.png",
  "media_type": "image",
  "mime_type": "image/png",
  "file_size": 12345,
  "sha256": "64-hex-chars",
  "status": "UPLOADED | PROCESSING | ANALYZED | FAILED | ARCHIVED",
  "analysis_status": "NONE | QUEUED | PROCESSING | COMPLETED | FAILED",
  "uploaded_by": 1,
  "created_at": "...",
  "storage_path": "cases/1/uuid.png",
  "notes": null
}
```

## 4. Analysis / Investigations

### `POST /api/evidence/{evidence_id}/analyze`

- **Auth:** Required (ownership check)
- **Body (JSON):**

```json
{
  "generate_explanation": true,
  "explainer": "gradcam",
  "verify_before_analyze": true,
  "explainers": ["gradcam","gradcam_plus_plus","layercam","scorecam","eigencam"],
  "advanced_xai": {
    "shap": true,
    "faithfulness": true,
    "counterfactual": true,
    "fusion": true,
    "trust": true,
    "multi_explainer": false
  }
}
```

- If `advanced_xai` is omitted or `null` → basic inference + basic explainer.
- `explainers` list can be `["gradcam"]` → single explainer; > 1 → multi-explainer.
- `advanced_xai` with expensive methods (SHAP, fusion, counterfactual, faithfulness, trust) run ONLY if opted-in.
- Explainability is best-effort: if the basic explainer or advanced XAI fails, the
  analysis still completes with the authenticity verdict, the XAI fields are left
  empty, and an `XAI_FAILED` audit event is recorded. Only an inference failure
  marks the run `FAILED` (`500 analysis_processing_error`).
- **201:** AnalysisRun object (below); status=COMPLETED or error=FAILED with error_message.

### `GET /api/analysis/{analysis_id}`

- **Auth:** Required (ownership check)
- Returns full analysis object.

### `GET /api/investigations/{analysis_id}`

- **Auth:** Required; alias for above (same result).

Analysis / Report Generation

### `POST /api/analysis/{analysis_id}/report`

- **Auth:** Required (analysis owner)
- **Body:** `{investigator_notes?:string, format?:"pdf"}`
- Generates PDF forensic report PDF.
- `format` is allowlisted (`pdf` only); any other value returns `400 validation_error`.
- **201:** Report object.

### `GET /api/analysis/{analysis_id}/reports`

- **Auth:** Required (ownership)
- List generated reports for this analysis.

### `GET /api/reports/{report_id}`

- **Auth:** Required (ownership)
- Report metadata.
- **404 `report_not_found`** when the report does not exist.

### `GET /api/reports/{report_id}/download`

- **Auth:** Required (ownership)
- Binary download of PDF. The served file must resolve inside `REPORT_DIR`;
  anything else returns `403 authorization_error`, and a missing file returns
  `404 report_file_missing`.

AnalysisRun object:

```json
{
  "analysis_id": 1,
  "investigation_id": "INV-2026-000001",
  "evidence_id": 1,
  "case_id": 1,
  "prediction": "REAL | FAKE",
  "confidence": 72.3,
  "analysis_status": "COMPLETED",
  "model_name": "efficientnet_b0",
  "model_version": "sprint3.2-best",
  "dataset_version": "v1",
  "artifact_dir": "/abs/path/to/artifacts/investigations/INV-..",
  "trust_score": 0.81,
  "quality_score": 0.75,
  "advanced_xai_results": {"methods_run":[...],"trust":{...},
  "error_message": null,
  "started_at":"...","completed_at":"...",
  "explanation": {"explainer":"gradcam","heatmap":"...","overlay":"...","explanation_json":"..."}
}
```

## 6. Audit (`/api/audit`)

### `GET /api/audit`

- **Auth:** Required
- ADMIN sees all rows; regular user only rows with their user_id.
- Limit 200 rows ordered by timestamp desc.

```json
{
  "audit_id": 5,
  "user_id": 1,
  "case_id": 1,
  "evidence_id": 1,
  "analysis_id": 1,
  "event_type": "ANALYSIS_COMPLETED",
  "timestamp: "2026-08-30T12:00:00",
  "details": {...}
}
```

Audit event types:

`USER_REGISTERED`, `USER_LOGIN`, `USER_LOGOUT`,
`CASE_CREATED`, `CASE_UPDATED`, `CASE_CLOSED`,
`EVIDENCE_UPLOADED`, `EVIDENCE_ACCESSED`, `EVIDENCE_VERIFIED`,
`ANALYSIS_STARTED`, `ANALYSIS_COMPLETED`, `ANALYSIS_FAILED`,
`XAI_GENERATED`, `XAI_FAILED`, `REPORT_GENERATED`,
`FACE_VERIFICATION_STARTED`, `FACE_VERIFICATION_COMPLETED`, `FACE_VERIFICATION_FAILED`

## 7. Face reference verification

### `POST /api/evidence/{evidence_id}/face-verification`

- **Auth:** Required (evidence/case ownership)
- **Body:** multipart `file` = reference image; optional form `threshold`, `no_match_threshold`, `investigation_id`
- **201:** Face verification object
- Decisions: `MATCH` | `NO_MATCH` | `INCONCLUSIVE`
- Rejects disallowed types/sizes using the evidence upload allowlist

### `GET /api/face-verifications/{verification_id}`

- **Auth:** Required (ownership)
- **200:** Face verification object

```json
{
  "verification_id": 1,
  "case_id": 1,
  "evidence_id": 1,
  "investigation_id": "INV-2026-000001",
  "verification_status": "COMPLETED",
  "decision": "MATCH",
  "similarity_score": 0.91,
  "distance_score": 0.42,
  "threshold": 0.70,
  "no_match_threshold": 0.50,
  "reason_code": "OK",
  "model_name": "inception_resnet_v1",
  "model_version": "vggface2",
  "reference_face_count": 1,
  "evidence_face_count": 1,
  "artifact_dir": "artifacts/investigations/INV-2026-000001/face_verification/…"
}
```

See [`PHASE5_FACE_VERIFICATION.md`](PHASE5_FACE_VERIFICATION.md).

## 8. Frontend support endpoints

Thin read-only routes added so the EVIDEX UI can render data that already
existed but had no HTTP surface. They add no business logic and reuse the same
ownership checks as the endpoints above.

### `GET /api/evidence/{evidence_id}/file`

- **Auth:** Required (evidence/case ownership)
- **200:** The stored evidence image inline (`Content-Type` = the recorded MIME type)
- Path resolved through `absolute_evidence_path`, so it cannot escape `UPLOAD_DIR`
- **404:** `not_found` when the stored file is missing

### `GET /api/evidence/{evidence_id}/analyses`

- **Auth:** Required (ownership)
- **200:** Array of analysis objects for that evidence, newest first
- Lets a reloaded page find an existing result, since `GET /api/analysis/{id}` needs an analysis id

### `GET /api/evidence/{evidence_id}/custody`

- **Auth:** Required (ownership)
- **200:** `{ evidence, case_number, case_title, events[], event_count }`
- `events` are `AuditLog` rows for that evidence in ascending time order
- No new custody table: this is a projection of the existing audit trail

### `GET /api/analysis/{analysis_id}/artifact/{kind}`

- **Auth:** Required (analysis ownership)
- **`kind`:** `heatmap` | `overlay` — anything else is `400 validation_error`
- **200:** The Grad-CAM PNG produced by the existing XAI stage
- **403:** `authorization_error` if the stored path resolves outside `artifacts/`
- **404:** `not_found` when the analysis produced no such artifact

### `GET /api/admin/users`

- **Auth:** Required, **ADMIN only** (`403 authorization_error` for any other role)
- **200:** every account from the `users` table via the standard user schema, each
  annotated with `case_count` and `evidence_count`
- Read-only projection; there is no create/update/suspend user endpoint

### `GET /api/dashboard/stats`

- **Auth:** Required. ADMIN sees all records (`scope: "all"`), others only their own (`scope: "own"`)
- **200:** counts, status tallies, `prediction` distribution, a real 14-day `trend`
  and 6-week `case_activity` series derived from stored timestamps, plus the
  10 most recent analyses
- Plain SQL aggregation; no model inference is involved

## 9. Static frontend

### `GET /evidex/` and `GET /evidex/{asset}`

Serves `DIGITALEVIDENCE_FIXED/` (the EVIDEX UI) from the Flask origin so the
Flask-Login session cookie stays first-party and no CORS configuration is
required. Only `index.html`, `api.js`, `script.js` and `styles.css` are
servable; every other path is `404`. The separate `frontend/` health shell
continues to own `/` and `/static`.

## 10. Error codes

| HTTP | error_code | Meaning |
|------|--------------|---------|
| 400 | bad_request | Malformed HTTP request |
| 400 | validation_error | Body/args invalid |
| 400 | invalid_evidence | Bad upload |
| 401 | authentication_error | Login required / bad credentials |
| 403 | authorization_error | Forbidden (ownership / role / path safety) |
| 404 | not_found / case_not_found / evidence_not_found / analysis_not_found / face_verification_not_found | Missing resource |
| 404 | report_not_found | Report row does not exist |
| 404 | report_file_missing | Report row exists but the stored file is gone |
| 405 | method_not_allowed | Wrong HTTP method for the route |
| 409 | conflict | Duplicate |
| 413 | payload_too_large | Upload exceeds `MAX_CONTENT_LENGTH` |
| 415 | unsupported_media_type | Unsupported request content type |
| 500 | analysis_processing_error | AI/XAI processing error |
| 500 | face_verification_processing_error | Face verification processing error |
| 500 | unhandled_error | Server error (safe message) |

All HTTP-level failures raised by the routing/parsing layer (405, 413, …) use the
same `{"ok": false, "error": ..., "message": ...}` envelope as service errors.
