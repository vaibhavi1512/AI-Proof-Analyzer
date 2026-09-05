# MAYA — Security Guide

## 1. Design-time controls

| Concern | Implementation |
|---------|---|
| Secrets | `.env` via `python-dotenv`; never in source. `ProductionConfig` refuses to boot on a placeholder or <16-char `SECRET_KEY` (`validate_production_config`) |
| Auth model | Flask-Login + Werkzeug argon2-family style `generate_password_hash` |
| Session cookie | `SESSION_COOKIE_HTTPONLY=true`, `SESSION_COOKIE_SAMESITE=Lax`; production adds `SESSION_COOKIE_SECURE=true` (opt-out via env) |
| Unhandled errors | `@app.errorhandler(Exception)` returns safe JSON envelope; stack trace stays server-side |
| ORM injection | SQLAlchemy 2.x parameterized queries only |
| AuthZ model | Owner (created_by_user_id) OR ADMIN — every case/evidence/analysis service function calls `require_case_access` or equivalent |
| Audit trail | Append-only `record_audit()` with secret scrubbing (password/token/secret/authorization keys stripped from `details` before DB insert) |
| File uploads | Extension + MIME + size ×2 (Werkzeug content-length, then post-save `stat().st_size`) |
| Filenames | UUID-based stored filename; never client filename used for paths directly |
| Path traversal | `backend/app/utils/paths.py` (`is_within` / `resolve_within`) compares resolved path components, not string prefixes, in evidence storage, evidence paths, face-reference temp files, and report downloads. Report downloads must resolve inside `REPORT_DIR` — never the repository root |
| Report filenames | Client `format` is allowlisted (`pdf`) before it reaches the output path |
| Hash computation | Server-computed SHA-256 via `ai.datasets.utils.checksums.hash_file`; client hashes ignored entirely |
| Configurable AI gates | Expensive SHAP/fusion/faithfulness/counterfactual only on opt-in `advanced_xai` booleans; DoS surface reduced |

## 2. Audit event model

`AuditLog` is append-only: there is **no** PATCH or DELETE route for audit rows through normal APIs. Admins can only list.

Event types: `USER_REGISTERED`, `USER_LOGIN`, `USER_LOGOUT`, `CASE_CREATED`, `CASE_UPDATED`, `CASE_CLOSED`, `EVIDENCE_UPLOADED`, `EVIDENCE_ACCESSED`, `EVIDENCE_VERIFIED`, `ANALYSIS_STARTED`, `ANALYSIS_COMPLETED`, `ANALYSIS_FAILED`, `XAI_GENERATED`, `XAI_FAILED`, `REPORT_GENERATED`, `FACE_VERIFICATION_STARTED`, `FACE_VERIFICATION_COMPLETED`, `FACE_VERIFICATION_FAILED`.

Secrets are redacted: `password`, `token`, `secret`, `authorization` (case-insensitive) are stripped from `details_json` server-side.

## 3. Hardened configurations

`ProductionConfig` disables `DEBUG`, marks cookies `Secure`, and fails fast on an unsafe `SECRET_KEY`. In the API, `handle_unhandled_exception` returns a fixed `"Internal server error"` message (never exception text). Werkzeug HTTP errors (405, 413, …) go through `handle_http_exception`, which preserves the real status code in the same JSON envelope instead of collapsing everything into a 500. For AI failures, the analysis service wraps low-level exceptions into `AnalysisProcessingError("Analysis processing failed. See logs for details.")` — only `error_type: str(type(exc).__name__)` is kept in the audit `details`.

Failed analyses and face verifications roll back partial writes before persisting their terminal `FAILED` state, so no row is left stranded in `PROCESSING`.

The Docker image and compose file no longer ship a default `SECRET_KEY`; `docker compose up` fails unless the operator provides one.

## 4. Known limitations (not certified)

> This code is NOT security-certified. It is a lab-grade investigation platform.

- No rate limiting on login/register (add `Flask-Limiter` for adversarial setups).
- SQLite default has no row-level ACLs; the service layer enforces them. Switch to Postgres + RLS if storing real-world PII.
- No CSRF tokens on JSON-only APIs (Flask-WTF CSRF disabled in TestingConfig; default dev config does not check JSON payloads). Protect with SameSite cookies + typical CORS/Origin headers in production.
- No signed URLs or expiring tokens for artifact download (current design uses cookie session auth only).
- Model checkpoint `best.pt` is trusted input; verify integrity out-of-band after transferring between machines.

## 5. Suggested deploy-time hardening (outside code)

- TLS-only reverse proxy with HSTS
- `ALLOW_PUBLIC_REGISTRATION=false` for operational deployments
- OS-level read-only mount for `artifacts/checkpoints/`
- Separate service account with minimum filesystem rights
- Antivirus/ML-safety scan on uploaded evidence before ingestion if operational deployment handles untrusted client payloads

## 6. Regression tests for security concerns

See `tests/test_e2e_product.py::test_e2e_authorization_enforced` (cross-user 403 on case/evidence/analysis/audit).

See `tests/test_backend_hardening.py` for the Phase 8 regressions: HTTP status
envelope (405/413/404), report download path traversal, `format` allowlisting,
`is_within` sibling-directory containment, non-fatal XAI, terminal `FAILED`
state, and production `SECRET_KEY` / cookie validation.
