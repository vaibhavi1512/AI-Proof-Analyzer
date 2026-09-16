/* ==========================================================================
   EVIDEX → MAYA backend API layer
   --------------------------------------------------------------------------
   The only place in the frontend that talks HTTP. Responsibilities:
     1. Centralised base URL (no hardcoded URLs in components).
     2. Flask-Login session cookie auth (same-origin, no tokens).
     3. Unwrap the backend {ok, data, message} envelope.
     4. Map real backend records onto the shapes the EVIDEX views expect.

   The backend is the source of truth. Nothing here invents forensic values:
   fields MAYA does not produce are returned as null and rendered by the UI as
   "not available" rather than filled with plausible-looking numbers.
   ========================================================================== */
(function (global) {
  'use strict';

  /* ---------------------------------------------------------------- config */
  // Served same-origin from Flask at /evidex/, so the API is a relative path.
  // Override before script.js loads (e.g. window.EVIDEX_API_BASE = 'http://127.0.0.1:5000')
  // when opening the frontend from a different origin.
  const API_BASE = (global.EVIDEX_API_BASE || '').replace(/\/$/, '');
  const url = (path) => `${API_BASE}${path}`;

  /* ---------------------------------------------------------------- errors */
  class ApiError extends Error {
    constructor(status, code, message) {
      super(message || 'Request failed');
      this.name = 'ApiError';
      this.status = status;
      this.code = code || 'error';
    }
    get isAuth() { return this.status === 401; }
    get isForbidden() { return this.status === 403; }
    get isNotFound() { return this.status === 404; }
  }

  // Human-facing text for the backend's status codes. Never converts a
  // failure into a success, and never hides the real status.
  const STATUS_TEXT = {
    400: 'The request was rejected as invalid.',
    401: 'Your session has expired. Please sign in again.',
    403: 'You are not authorised to access this resource.',
    404: 'The requested resource was not found.',
    405: 'That operation is not supported on this resource.',
    413: 'The file is larger than the server upload limit.',
    415: 'That file type is not supported.',
    422: 'The request could not be processed.',
    500: 'The server encountered an internal error.',
  };

  async function request(path, options = {}) {
    const opts = {
      method: options.method || 'GET',
      // Session cookie must ride along on every call.
      credentials: 'include',
      headers: options.headers || {},
    };
    if (options.json !== undefined) {
      opts.headers = { ...opts.headers, 'Content-Type': 'application/json' };
      opts.body = JSON.stringify(options.json);
    } else if (options.body !== undefined) {
      // FormData: let the browser set the multipart boundary.
      opts.body = options.body;
    }

    let res;
    try {
      res = await fetch(url(path), opts);
    } catch (networkErr) {
      throw new ApiError(0, 'network_error',
        'Cannot reach the MAYA backend. Is the server running?');
    }

    if (res.status === 204) return null;

    const contentType = res.headers.get('Content-Type') || '';
    if (!contentType.includes('application/json')) {
      if (!res.ok) {
        throw new ApiError(res.status, 'http_error',
          STATUS_TEXT[res.status] || `Request failed (${res.status}).`);
      }
      return res;
    }

    const body = await res.json().catch(() => null);
    if (!res.ok || (body && body.ok === false)) {
      const code = (body && body.error) || 'http_error';
      const message = (body && body.message) || STATUS_TEXT[res.status] ||
        `Request failed (${res.status}).`;
      throw new ApiError(res.status, code, message);
    }
    return body ? body.data : null;
  }

  /* ------------------------------------------------------------- endpoints */
  const auth = {
    register: (payload) => request('/api/auth/register', { method: 'POST', json: payload }),
    login: (login, password) => request('/api/auth/login', { method: 'POST', json: { login, password } }),
    logout: () => request('/api/auth/logout', { method: 'POST' }),
    me: () => request('/api/auth/me'),
  };

  const cases = {
    list: () => request('/api/cases'),
    create: (payload) => request('/api/cases', { method: 'POST', json: payload }),
    get: (id) => request(`/api/cases/${id}`),
    close: (id) => request(`/api/cases/${id}/close`, { method: 'POST' }),
    remove: (id) => request(`/api/cases/${id}`, { method: 'DELETE' }),
  };

  const evidence = {
    listByCase: (caseId) => request(`/api/evidence/cases/${caseId}`),
    get: (id) => request(`/api/evidence/${id}`),
    verifyIntegrity: (id) => request(`/api/evidence/${id}/verify-integrity`, { method: 'POST' }),
    custody: (id) => request(`/api/evidence/${id}/custody`),
    analyses: (id) => request(`/api/evidence/${id}/analyses`),
    fileUrl: (id) => url(`/api/evidence/${id}/file`),
    upload: (caseId, file, notes) => {
      const form = new FormData();
      form.append('file', file);
      if (notes) form.append('notes', notes);
      return request(`/api/evidence/cases/${caseId}`, { method: 'POST', body: form });
    },
  };

  const analysis = {
    analyze: (evidenceId, opts = {}) => request(`/api/evidence/${evidenceId}/analyze`, {
      method: 'POST',
      json: {
        generate_explanation: opts.generateExplanation !== false,
        explainer: opts.explainer || 'gradcam',
        verify_before_analyze: opts.verifyBeforeAnalyze !== false,
        ...(opts.advancedXai ? { advanced_xai: opts.advancedXai } : {}),
      },
    }),
    get: (id) => request(`/api/analysis/${id}`),
    // Real Grad-CAM images produced by the existing XAI stage.
    artifactUrl: (id, kind) => url(`/api/analysis/${id}/artifact/${kind}`),
    listReports: (id) => request(`/api/analysis/${id}/reports`),
    generateReport: (id, notes) => request(`/api/analysis/${id}/report`, {
      method: 'POST',
      json: notes ? { investigator_notes: notes } : {},
    }),
  };

  const reports = {
    get: (id) => request(`/api/reports/${id}`),
    downloadUrl: (id) => url(`/api/reports/${id}/download`),
  };

  const faceVerification = {
    create: (evidenceId, file, opts = {}) => {
      const form = new FormData();
      // Backend expects the multipart field to be named 'file'.
      form.append('file', file);
      if (opts.threshold !== undefined && opts.threshold !== null) {
        form.append('threshold', String(opts.threshold));
      }
      if (opts.investigationId) form.append('investigation_id', opts.investigationId);
      return request(`/api/evidence/${evidenceId}/face-verification`, { method: 'POST', body: form });
    },
    get: (id) => request(`/api/face-verifications/${id}`),
  };

  const audit = { list: () => request('/api/audit') };
  const dashboard = { stats: () => request('/api/dashboard/stats') };
  // ADMIN-only read of the real user table; the backend rejects other roles.
  const admin = { users: () => request('/api/admin/users') };

  /* -------------------------------------------------------------- adapters
     Translate backend records into the field names the existing EVIDEX views
     already read, so the views themselves stay untouched.
     ---------------------------------------------------------------------- */

  // MAYA is a binary REAL/FAKE classifier. EVIDEX was written around a
  // four-state verdict; we map only what the model actually supports and mark
  // the rest unavailable instead of inventing "Suspicious"/"Inconclusive".
  function verdictFromPrediction(prediction, confidence) {
    if (prediction === 'REAL') return 'Authentic';
    if (prediction === 'FAKE') return 'Tampered';
    return 'Unavailable';
  }

  function riskFromPrediction(prediction, confidence) {
    if (prediction === 'FAKE') return confidence >= 70 ? 'High' : 'Medium';
    if (prediction === 'REAL') return confidence >= 70 ? 'Low' : 'Medium';
    return 'Unknown';
  }

  function kindFromMime(mime, filename) {
    const m = (mime || '').toLowerCase();
    if (m.startsWith('image/')) return 'Image';
    if (m.startsWith('video/')) return 'Video';
    if (m.startsWith('audio/')) return 'Audio';
    const ext = (filename || '').split('.').pop().toLowerCase();
    if (['jpg', 'jpeg', 'png', 'bmp', 'webp'].includes(ext)) return 'Image';
    return 'Document';
  }

  function adaptCase(c) {
    return {
      id: String(c.case_id ?? c.id),
      backendId: c.case_id ?? c.id,
      caseNumber: c.case_number,
      name: c.title,
      description: c.description || '',
      status: titleCaseStatus(c.status),
      priority: c.priority,
      opened: c.created_at,
      updated: c.updated_at || c.created_at,
      closedAt: c.closed_at,
      createdBy: c.created_by,
      evidenceIds: [],
      // Derived from the case's evidence once analyses are loaded.
      risk: 'Unknown',
      investigator: null,
    };
  }

  function titleCaseStatus(status) {
    switch (status) {
      case 'OPEN': return 'Open';
      case 'IN_PROGRESS': return 'Under Review';
      case 'CLOSED': return 'Closed';
      case 'ARCHIVED': return 'Archived';
      default: return status || 'Unknown';
    }
  }

  function adaptEvidence(e, caseRef) {
    return {
      id: String(e.evidence_id),
      backendId: e.evidence_id,
      caseId: caseRef ? String(caseRef) : String(e.case_id),
      caseBackendId: e.case_id,
      filename: e.original_filename,
      type: kindFromMime(e.mime_type, e.original_filename),
      mimeType: e.mime_type,
      size: e.file_size,
      // Server-computed SHA-256 — authoritative, never recomputed client-side.
      sha256: e.sha256,
      timestamp: e.created_at,
      evidenceStatus: e.status,
      analysisStatus: e.analysis_status,
      notes: e.notes,
      uploadedBy: e.uploaded_by,
      // Populated from a real AnalysisRun when one exists.
      score: null,
      status: 'Not analysed',
      risk: 'Unknown',
      investigator: null,
      analysisId: null,
    };
  }

  // Merge a real analysis onto its evidence record.
  function applyAnalysis(ev, run) {
    if (!run || run.analysis_status !== 'COMPLETED') {
      ev.analysisId = run ? run.analysis_id : null;
      ev.status = run && run.analysis_status === 'FAILED' ? 'Analysis failed' : ev.status;
      return ev;
    }
    const confidence = typeof run.confidence === 'number' ? run.confidence : null;
    ev.analysisId = run.analysis_id;
    ev.investigationId = run.investigation_id;
    ev.prediction = run.prediction;
    ev.confidence = confidence;
    ev.realProbability = typeof run.real_probability === 'number' ? run.real_probability : null;
    ev.fakeProbability = typeof run.fake_probability === 'number' ? run.fake_probability : null;
    ev.score = confidence;
    ev.status = verdictFromPrediction(run.prediction, confidence);
    ev.risk = riskFromPrediction(run.prediction, confidence);
    ev.modelName = run.model_name;
    ev.modelVersion = run.model_version;
    ev.trustScore = run.trust_score;
    ev.qualityScore = run.quality_score;
    ev.hasHeatmap = !!(run.explanation && run.explanation.heatmap);
    ev.hasOverlay = !!(run.explanation && run.explanation.overlay);
    ev.explainer = run.explanation ? run.explanation.explainer : null;
    ev.advancedXai = run.advanced_xai_results || null;
    ev.completedAt = run.completed_at;
    return ev;
  }

  function adaptAnalysis(run) {
    const confidence = typeof run.confidence === 'number' ? run.confidence : null;
    return {
      analysisId: run.analysis_id,
      investigationId: run.investigation_id,
      evidenceId: run.evidence_id,
      caseId: run.case_id,
      prediction: run.prediction,
      confidence,
      realProbability: typeof run.real_probability === 'number' ? run.real_probability : null,
      fakeProbability: typeof run.fake_probability === 'number' ? run.fake_probability : null,
      status: verdictFromPrediction(run.prediction, confidence),
      risk: riskFromPrediction(run.prediction, confidence),
      analysisStatus: run.analysis_status,
      modelName: run.model_name,
      modelVersion: run.model_version,
      datasetVersion: run.dataset_version,
      trustScore: run.trust_score,
      qualityScore: run.quality_score,
      explanation: run.explanation || null,
      advancedXai: run.advanced_xai_results || null,
      errorMessage: run.error_message,
      startedAt: run.started_at,
      completedAt: run.completed_at,
      artifactDir: run.artifact_dir,
    };
  }

  function adaptReport(r) {
    return {
      reportId: r.report_id,
      reportNumber: r.report_number,
      caseId: r.case_id,
      evidenceId: r.evidence_id,
      analysisId: r.analysis_id,
      investigationId: r.investigation_id,
      format: r.format,
      sizeBytes: r.size_bytes,
      sha256: r.sha256,
      generatedAt: r.generated_at,
      notes: r.investigator_notes,
    };
  }

  // Audit event type -> the custody categories the EVIDEX timeline renders.
  const EVENT_CATEGORY = {
    USER_REGISTERED: 'AUTH', USER_LOGIN: 'AUTH', USER_LOGOUT: 'AUTH',
    CASE_CREATED: 'CASE', CASE_UPDATED: 'CASE', CASE_CLOSED: 'CASE',
    EVIDENCE_UPLOADED: 'UPLOAD', EVIDENCE_ACCESSED: 'CUSTODY',
    EVIDENCE_VERIFIED: 'INTEGRITY',
    ANALYSIS_STARTED: 'ANALYSIS', ANALYSIS_COMPLETED: 'ANALYSIS',
    ANALYSIS_FAILED: 'ANALYSIS',
    XAI_GENERATED: 'ANALYSIS', XAI_FAILED: 'ANALYSIS',
    REPORT_GENERATED: 'REPORT',
    FACE_VERIFICATION_STARTED: 'FACE', FACE_VERIFICATION_COMPLETED: 'FACE',
    FACE_VERIFICATION_FAILED: 'FACE',
  };

  const EVENT_LABEL = {
    EVIDENCE_UPLOADED: 'Evidence uploaded and hashed',
    EVIDENCE_VERIFIED: 'Integrity re-verified (SHA-256)',
    EVIDENCE_ACCESSED: 'Evidence accessed',
    ANALYSIS_STARTED: 'Authenticity analysis started',
    ANALYSIS_COMPLETED: 'Authenticity analysis completed',
    ANALYSIS_FAILED: 'Authenticity analysis failed',
    XAI_GENERATED: 'Explainability (Grad-CAM) generated',
    XAI_FAILED: 'Explainability generation failed',
    REPORT_GENERATED: 'Forensic report generated',
    FACE_VERIFICATION_STARTED: 'Face reference verification started',
    FACE_VERIFICATION_COMPLETED: 'Face reference verification completed',
    FACE_VERIFICATION_FAILED: 'Face reference verification failed',
    CASE_CREATED: 'Case opened',
    CASE_UPDATED: 'Case updated',
    CASE_CLOSED: 'Case closed',
  };

  function adaptAuditEvent(e) {
    return {
      id: 'LOG-' + e.audit_id,
      type: EVENT_CATEGORY[e.event_type] || 'SYSTEM',
      eventType: e.event_type,
      message: EVENT_LABEL[e.event_type] || e.event_type.replace(/_/g, ' ').toLowerCase(),
      actor: e.user_id ? 'user#' + e.user_id : 'system',
      actorId: e.user_id,
      time: e.timestamp,
      caseId: e.case_id,
      evidenceId: e.evidence_id,
      analysisId: e.analysis_id,
      details: e.details || {},
      severity: /FAILED/.test(e.event_type) ? 'warning' : 'success',
    };
  }

  global.MayaApi = {
    ApiError, request, API_BASE,
    baseUrl: API_BASE,
    auth, cases, evidence, analysis, reports, faceVerification, audit, dashboard, admin,
    adapt: {
      case: adaptCase, evidence: adaptEvidence, analysis: adaptAnalysis,
      report: adaptReport, auditEvent: adaptAuditEvent, applyAnalysis,
      verdictFromPrediction, riskFromPrediction, titleCaseStatus, kindFromMime,
    },
  };
})(window);
