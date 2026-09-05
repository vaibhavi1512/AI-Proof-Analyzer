# MAYA — Testing Guide

## Test matrix

| Category | File | Covers | Uses real AI |
|---|---|---|---|
| Auth | `tests/test_auth_api.py` | Register, login, logout, me, duplicate, roles, no pw leak | No |
| Cases | `tests/test_cases_api.py` | CRUD, close, ownership, ADMIN visibility via e2e | No |
| Evidence | `tests/test_evidence_api.py` | Upload, SHA-256, integrity, reject bad files, ownership | No |
| Analysis (mocked) | `tests/test_analysis_api.py` | Product flow with mocked inference/XAI; failure persistence | No (patched) |
| **Face verification** | `tests/test_face_verification.py` | MATCH/NO_MATCH/INCONCLUSIVE, authz, artifacts, audit, report integration (engine mocked) | No (patched) |
| **Backend hardening** | `tests/test_backend_hardening.py` | Phase 8 regressions: 405/413/404 envelopes, report download traversal + `format` allowlist, path containment, non-fatal XAI, terminal FAILED state, production `SECRET_KEY`/cookies | No (patched) |
| **E2E Product** | `tests/test_e2e_product.py` | **Register → Login → Case → Upload → Real EfficientNet inference → Audit → Integrity → Artifacts → Cross-user authZ → Tamper detection** | **YES** |
| Audit (XAI internal) | `tests/test_audit.py` | Phase 4.5 `build_audit_record` / `XaiAuditRecord` serialization | No |
| AI unit suite | `tests/test_inference.py`, `test_model.py`, `test_training.py`, `test_evaluation.py`, `test_dataset.py` | Training/inference correctness | Optional |
| XAI suite | `tests/test_gradcam.py`, `test_multi_explainer.py`, `test_explanation_analytics.py`, `test_explainability_benchmark.py`, `test_shap.py`, `test_faithfulness.py`, `test_counterfactual.py`, `test_fusion.py`, `test_trust.py`, `test_xai_stabilization.py` | XAI algorithms + benchmarks offline | Optional |

## Running tests

### Fast product subset (no torch required, 30-60 s typical on 8 GB Windows 11)

```powershell
python -m pytest tests/test_auth_api.py tests/test_cases_api.py `
  tests/test_evidence_api.py tests/test_audit.py `
  tests/test_analysis_api.py tests/test_face_verification.py `
  tests/test_backend_hardening.py -q
```

### Real AI + full E2E backend

```powershell
# requires torch + torchvision + best.pt checkpoint in artifacts/checkpoints/
python -m pytest tests/test_e2e_product.py -v --tb=short
```

### Full suite (ALL tests — product + AI + XAI)

```powershell
python -m pytest tests/ -q
```

### Fixtures

Shared fixtures live in `tests/conftest_product.py`:
- `app(tmp_path)`: `create_app("testing")` with isolated in-memory DB and `tmp_path/uploads`
- `client(app)`: `app.test_client()`
- `register_and_login(client, username, email)`: 2 API calls, returns user dict with id+role

## Adding new tests

1. For product API tests → prefer real `client()` calls with app fixture.
2. Mock only where the alternative would require external network or > 1 minute CPU.
3. A genuine non-mocked integration test MUST exist that exercises `InferencePipeline.run` end-to-end through the product API (see `test_e2e_auth_case_evidence_analysis`).
4. Never weaken assertions to make a flaky test pass; diagnose and fix the flake.

## Diagnostic script (manual)

`scripts/_diag_product_ai.py` runs: Register → Login → Case → Upload → Real inference (no XAI) → Integrity verify → Audit check. Use this as a smoke test when debugging infrastructure.
