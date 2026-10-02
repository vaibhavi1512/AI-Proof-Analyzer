/**
 * Video-analysis result markup for the 16-frame LSTM.
 * Image results do not use this module. Missing fields stay blank.
 */
(function (root) {
  "use strict";

  function escapeHtml(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function finiteNumber(value) {
    return typeof value === "number" && Number.isFinite(value) ? value : null;
  }

  function formatFakeProbability(value) {
    const number = finiteNumber(value);
    if (number === null) return null;
    return (number * 100).toFixed(2) + "%";
  }

  function predictionLabel(value) {
    return value === "REAL" || value === "FAKE" ? value : null;
  }

  function readPFake(video) {
    if (!video) return null;
    const adapted = finiteNumber(video.pFake);
    if (adapted !== null) return adapted;
    return finiteNumber(video.p_fake);
  }

  /**
   * Probability of the predicted class, as a percentage.
   * FAKE → p_fake × 100. REAL → (1 − p_fake) × 100.
   */
  function confidenceScore(video) {
    const prediction = predictionLabel(video && video.prediction);
    const pFake = readPFake(video);
    if (!prediction || pFake === null) return null;
    return (prediction === "FAKE" ? pFake : (1 - pFake)) * 100;
  }

  function formatConfidenceScore(video) {
    const score = confidenceScore(video);
    return score === null ? null : score.toFixed(2) + "%";
  }

  function countLabel(value) {
    const number = finiteNumber(value);
    return number === null ? null : String(Math.round(number));
  }

  function frameRows(video) {
    if (!video) return [];
    if (video.xai && Array.isArray(video.xai.frames) && video.xai.frames.length) {
      return video.xai.frames;
    }
    return Array.isArray(video.frames) ? video.frames : [];
  }

  function temporalPoints(video) {
    return frameRows(video)
      .filter(function (frame) { return finiteNumber(frame.temporal_importance) !== null; })
      .map(function (frame) {
        return {
          frameIndex: frame.frame_index,
          timestampSeconds: finiteNumber(frame.timestamp_seconds),
          contribution: frame.temporal_importance,
          fallback: frame.used_full_frame_fallback === true,
        };
      });
  }

  function fallbackMessage(count) {
    if (count === 1) {
      return "1 sampled frame had no detectable face and was analyzed using the full-frame fallback.";
    }
    if (count !== null && count > 1) {
      return "Frames without a detectable face were analyzed using the full-frame fallback.";
    }
    return "";
  }

  function inputKind(frame) {
    if (frame.used_full_frame_fallback === true) return "full-frame fallback";
    if (frame.used_face_crop === true) return "face crop";
    return null;
  }

  function videoProgressSteps() {
    return [
      "Integrity verification (SHA-256)",
      "16-frame LSTM inference",
      "Optional video XAI",
      "Persisting investigation artifacts",
    ];
  }

  function artifactSrc(artifactUrl, analysisId, kind, frameIndex) {
    if (analysisId == null || typeof artifactUrl !== "function") return "";
    return artifactUrl(analysisId, kind, frameIndex);
  }

  function gradcamEntries(video) {
    const byIndex = new Map();
    frameRows(video).forEach(function (frame) {
      const index = Number(frame.frame_index);
      if (Number.isInteger(index)) byIndex.set(index, frame);
    });
    return (video.gradcamFrameIndices || [])
      .map(function (value) { return Number(value); })
      .filter(function (index) { return Number.isInteger(index); })
      .map(function (index) {
        return byIndex.get(index) || { frame_index: index };
      });
  }

  function temporalBody(video) {
    if (!video || video.xaiAvailable !== true) {
      return `<p class="video-muted">XAI unavailable.</p>`;
    }
    if (temporalPoints(video).length === 0) {
      return `<p class="video-missing-artifact">Temporal explanation is not available for this analysis.</p>`;
    }
    return `<canvas id="videoTemporalChart" aria-label="Relative temporal contribution"></canvas>`;
  }

  function predictionBanner(video) {
    const prediction = predictionLabel(video && video.prediction);
    const confidence = formatConfidenceScore(video);
    const tone = prediction === "FAKE" ? "red" : prediction === "REAL" ? "green" : "blue";
    return `
      <div class="status-banner video-result-banner" style="border-color:var(--${tone});background:var(--${tone}-dim);">
        <div>
          <p class="video-kicker">Prediction</p>
          <h3 style="color:var(--${tone});">${prediction ? escapeHtml(prediction) : "—"}</h3>
        </div>
        <div>
          <p class="video-kicker">Confidence Score</p>
          <h3 data-field="confidence" style="color:var(--${tone});">${confidence ? escapeHtml(confidence) : "—"}</h3>
        </div>
      </div>`;
  }

  function renderVideoFailure(message) {
    const detail = message ? escapeHtml(message) : "The video analysis did not complete.";
    return `
      <section class="video-analysis" data-video-state="failed">
        <div class="card">
          <div class="panel-title"><h3>VIDEO ANALYSIS</h3></div>
          <p class="video-status-error">Analysis failed. ${detail}</p>
        </div>
      </section>`;
  }

  function beginNewAnalysis() {
    return `
      <section class="video-analysis" data-video-state="pending" data-analysis-id="">
        <p class="video-muted">Video analysis in progress.</p>
      </section>`;
  }

  function renderVideoAnalysisSection(analysis, options) {
    const video = analysis && analysis.videoAnalysis;
    if (!video) return "";
    const opts = options || {};
    const analysisId = opts.analysisId;
    const framesAnalyzed = countLabel(video.framesAnalyzed);
    const faceCrops = countLabel(video.faceCropFrames);
    const fallback = finiteNumber(video.fallbackFrames);
    const fallbackText = countLabel(video.fallbackFrames);
    const temporalWording = (video.xai && video.xai.temporalWording) || "frames with higher relative contribution";

    return `
      <section class="video-analysis" data-video-screen="analysis" data-video-state="completed" data-analysis-id="${escapeHtml(analysisId == null ? "" : analysisId)}">
        <div class="card">
          <div class="panel-title"><h3>VIDEO ANALYSIS</h3><span class="sub">16-frame LSTM</span></div>
          ${predictionBanner(video)}
          <div class="kv-grid video-metrics">
            <div class="kv-item"><div class="kl">Frames analyzed</div><div class="kv-val" data-field="frames">${framesAnalyzed ? escapeHtml(framesAnalyzed) : "—"}</div></div>
            <div class="kv-item"><div class="kl">Face-crop frames</div><div class="kv-val" data-field="face-crops">${faceCrops ? escapeHtml(faceCrops) : "—"}</div></div>
            <div class="kv-item"><div class="kl">Full-frame fallback</div><div class="kv-val" data-field="fallback">${fallbackText ? escapeHtml(fallbackText) : "—"}</div></div>
            <div class="kv-item"><div class="kl">Model</div><div class="kv-val">${escapeHtml(video.modelName || "16-frame LSTM")}</div></div>
            <div class="kv-item"><div class="kl">Model version</div><div class="kv-val">${escapeHtml(video.modelVersion || "—")}</div></div>
          </div>
          <p class="video-meaning">Confidence Score is the model-assigned probability of the predicted class. It is not proof.</p>
          ${fallback !== null && fallback > 0 ? `<p class="video-fallback-note">${escapeHtml(fallbackMessage(fallback))}</p>` : ""}
        </div>
        <div class="card" data-temporal-xai>
          <div class="panel-title"><h3>Temporal Analysis</h3><span class="sub">Relative temporal contribution</span></div>
          ${video.xaiAvailable === true ? `<p class="video-meaning">${escapeHtml(temporalWording)}. Higher values indicate frames that contributed more strongly to the model prediction.</p>` : ""}
          ${temporalBody(video)}
        </div>
        <div class="card">
          <div class="panel-title"><h3>Model and evidence</h3></div>
          <div class="kv-grid">
            <div class="kv-item"><div class="kl">Model identifier</div><div class="kv-val">${escapeHtml(video.modelName || "—")}</div></div>
            <div class="kv-item"><div class="kl">Model version</div><div class="kv-val">${escapeHtml(video.modelVersion || "—")}</div></div>
            <div class="kv-item"><div class="kl">Evidence</div><div class="kv-val">${escapeHtml(opts.evidenceLabel || "—")}</div></div>
            <div class="kv-item"><div class="kl">SHA-256</div><div class="kv-val">${escapeHtml(opts.sha256 || "—")}</div></div>
          </div>
          ${analysisId == null ? "" : `<button class="btn btn-primary btn-sm" data-action="generate-report" data-analysis="${escapeHtml(analysisId)}" style="margin-top:14px;">Generate Report</button>`}
        </div>
      </section>`;
  }

  function xaiPreview(video, analysisId, artifactUrl) {
    if (!video || video.xaiAvailable !== true) {
      return `<p class="video-muted">XAI unavailable.</p>`;
    }
    const spatial = (video.xai && video.xai.spatialWording) || "regions contributing to the model prediction";
    const temporal = (video.xai && video.xai.temporalWording) || "frames with higher relative contribution";
    const entries = gradcamEntries(video);
    const contact = video.gradcamContactSheet === true
      ? artifactSrc(artifactUrl, analysisId, "video_contact_sheet")
      : "";
    const preview = entries[0]
      ? artifactSrc(artifactUrl, analysisId, "video_gradcam", entries[0].frame_index)
      : "";
    const count = entries.length;
    return `
      <p class="video-meaning">Relative temporal contribution: ${escapeHtml(temporal)}. Higher values indicate frames that contributed more strongly to the model prediction.</p>
      <p class="video-meaning">Spatial explanation summary: ${escapeHtml(spatial)}. These regions explain model behavior. They are not confirmed manipulated pixels.</p>
      ${count ? `<p class="video-meaning">Grad-CAM preview covers ${count} frame${count === 1 ? "" : "s"}. Open Tampering Map to inspect each frame.</p>` : ""}
      ${contact ? `<img class="video-contact-sheet" alt="Regions contributing to the model prediction" src="${escapeHtml(contact)}">` : ""}
      ${preview ? `<img class="video-gradcam video-xai-preview" alt="Regions contributing to the model prediction" src="${escapeHtml(preview)}">` : ""}
      ${!contact && !preview ? `<p class="video-missing-artifact">Grad-CAM artifact is not available for this analysis.</p>` : ""}`;
  }

  function renderVideoExplainability(analysis, options) {
    const video = analysis && analysis.videoAnalysis;
    if (!video) return "";
    const opts = options || {};
    const artifactUrl = typeof opts.artifactUrl === "function" ? opts.artifactUrl : function () { return ""; };
    const modelName = video.modelName || (analysis && analysis.modelName) || "—";
    const modelVersion = video.modelVersion || (analysis && analysis.modelVersion) || "—";
    const available = video.xaiAvailable === true;
    return `
      <section class="video-analysis" data-video-screen="xai" data-video-state="completed" data-analysis-id="${escapeHtml(opts.analysisId == null ? "" : opts.analysisId)}">
        <div class="card">
          <div class="panel-title"><h3>XAI Insights</h3><span class="sub">${escapeHtml(modelName)}</span></div>
          ${predictionBanner(video)}
          <div class="kv-grid video-metrics">
            <div class="kv-item"><div class="kl">Model</div><div class="kv-val">${escapeHtml(modelName)}</div></div>
            <div class="kv-item"><div class="kl">Model version</div><div class="kv-val">${escapeHtml(modelVersion)}</div></div>
          </div>
          <p class="video-meaning">Confidence Score is the model-assigned probability of the predicted class. It is not proof.</p>
        </div>
        ${available ? `
        <div class="card" data-temporal-xai>
          <div class="panel-title"><h3>Temporal Analysis</h3><span class="sub">Relative temporal contribution</span></div>
          ${temporalBody(video)}
        </div>
        <div class="card" data-spatial-summary>
          <div class="panel-title"><h3>Regions contributing to the model prediction</h3></div>
          ${xaiPreview(video, opts.analysisId, artifactUrl)}
        </div>` : `
        <div class="card" data-xai-unavailable>
          <p class="video-muted">XAI unavailable.</p>
        </div>`}
      </section>`;
  }

  function videoRecord(run) {
    if (!run || typeof run !== "object") return null;
    if (run.video_analysis && typeof run.video_analysis === "object") return run.video_analysis;
    if (run.videoAnalysis && typeof run.videoAnalysis === "object") return run.videoAnalysis;
    return null;
  }

  function gradcamIndices(video) {
    const list = video.gradcamFrameIndices || video.gradcam_frame_indices;
    if (!Array.isArray(list)) return [];
    return list.filter(function (value) { return Number.isInteger(value); });
  }

  /**
   * Video-only readiness facts. Returns null when video_analysis is absent
   * so the existing image checklist keeps using heatmap, overlay, and confidence.
   */
  function evidenceReadiness(run) {
    const video = videoRecord(run);
    if (!video) return null;
    const prediction = predictionLabel(video.prediction);
    const score = formatConfidenceScore(video);
    const model = video.model_name || video.modelName || (run && run.model_name) || "—";
    const version = video.model_version || video.modelVersion || (run && run.model_version) || "—";
    const analysisNote = !prediction
      ? "No completed analysis"
      : (score
        ? prediction + " at " + score + " confidence (" + model + " " + version + ")"
        : prediction + " (confidence not reported) (" + model + " " + version + ")");
    const xaiOn = video.xaiAvailable === true || video.xai_available === true;
    const contact = video.gradcamContactSheet === true || video.gradcam_contact_sheet === true;
    const spatial = contact || gradcamIndices(video).length > 0;
    const temporal = temporalPoints(video).length > 0;
    const explainabilityOk = xaiOn && (spatial || temporal);
    let explainabilityNote = "XAI unavailable";
    if (explainabilityOk && spatial) explainabilityNote = "Grad-CAM available for this video analysis";
    else if (explainabilityOk) explainabilityNote = "Relative temporal contribution available";
    return {
      analysisNote: analysisNote,
      explainabilityOk: explainabilityOk,
      explainabilityNote: explainabilityNote,
    };
  }

  function frameMeta(frame) {
    const kind = inputKind(frame);
    const time = finiteNumber(frame.timestamp_seconds);
    return "Frame " + frame.frame_index
      + (time === null ? "" : " · " + time.toFixed(2) + "s")
      + (kind ? " · " + kind : "");
  }

  function renderVideoAttentionMap(analysis, options) {
    const video = analysis && analysis.videoAnalysis;
    if (!video) return "";
    const opts = options || {};
    const artifactUrl = typeof opts.artifactUrl === "function" ? opts.artifactUrl : function () { return ""; };
    const modelName = video.modelName || (analysis && analysis.modelName) || "—";
    const modelVersion = video.modelVersion || (analysis && analysis.modelVersion) || "—";
    const entries = gradcamEntries(video);
    const requested = Number(opts.selectedFrame);
    const selected = entries.find(function (frame) { return frame.frame_index === requested; }) || entries[0] || null;
    const contact = video.gradcamContactSheet === true
      ? artifactSrc(artifactUrl, opts.analysisId, "video_contact_sheet")
      : "";
    const selectedSrc = selected
      ? artifactSrc(artifactUrl, opts.analysisId, "video_gradcam", selected.frame_index)
      : "";
    let body;
    if (video.xaiAvailable !== true) {
      body = `<p class="video-muted">XAI unavailable.</p>`;
    } else if (!contact && !selectedSrc) {
      body = `<p class="video-missing-artifact">Grad-CAM artifact is not available for this analysis.</p>`;
    } else {
      const buttons = entries.map(function (frame) {
        const src = artifactSrc(artifactUrl, opts.analysisId, "video_gradcam", frame.frame_index);
        const meta = frameMeta(frame);
        const active = selected && frame.frame_index === selected.frame_index;
        return `<button type="button" class="heatmap-tab${active ? " active" : ""}" data-video-frame="${escapeHtml(frame.frame_index)}" data-src="${escapeHtml(src)}" data-meta="${escapeHtml(meta)}">${escapeHtml(meta)}</button>`;
      }).join("");
      body = `
        <p class="video-meaning">Regions contributing to the model prediction. These regions explain model behavior. They are not confirmed manipulated pixels.</p>
        ${contact ? `<img class="video-contact-sheet" alt="Regions contributing to the model prediction" src="${escapeHtml(contact)}">` : ""}
        ${buttons ? `<div class="video-frame-switch" role="group" aria-label="Grad-CAM frames">${buttons}</div>` : ""}
        ${selectedSrc ? `
          <img class="video-gradcam" data-tamper-frame alt="${escapeHtml(frameMeta(selected))}" src="${escapeHtml(selectedSrc)}">
          <p class="video-frame-meta" data-tamper-meta>${escapeHtml(frameMeta(selected))}</p>
        ` : ""}`;
    }
    return `
      <section class="video-analysis" data-video-screen="attention" data-video-state="completed" data-analysis-id="${escapeHtml(opts.analysisId == null ? "" : opts.analysisId)}">
        <div class="card">
          <div class="panel-title"><h3>Tampering Map</h3><span class="sub">${escapeHtml(modelName)} ${escapeHtml(modelVersion)}</span></div>
          ${predictionBanner(video)}
        </div>
        <div class="card" data-spatial-xai>
          <div class="panel-title"><h3>Regions contributing to the model prediction</h3></div>
          ${body}
        </div>
      </section>`;
  }

  function bindTamperingMap(root) {
    const scope = root && root.querySelector ? root : null;
    const section = scope
      ? scope.querySelector('[data-video-screen="attention"]')
      : null;
    if (!section) return;
    const image = section.querySelector("[data-tamper-frame]");
    const meta = section.querySelector("[data-tamper-meta]");
    section.querySelectorAll("[data-video-frame]").forEach(function (button) {
      button.addEventListener("click", function () {
        section.querySelectorAll("[data-video-frame]").forEach(function (other) {
          other.classList.toggle("active", other === button);
        });
        const src = button.getAttribute("data-src");
        if (image && src) image.src = src;
        if (meta) meta.textContent = button.getAttribute("data-meta") || "";
      });
    });
  }

  root.MayaVideoResult = {
    videoProgressSteps: videoProgressSteps,
    formatFakeProbability: formatFakeProbability,
    formatConfidenceScore: formatConfidenceScore,
    confidenceScore: confidenceScore,
    temporalPoints: temporalPoints,
    beginNewAnalysis: beginNewAnalysis,
    renderVideoFailure: renderVideoFailure,
    renderVideoAnalysisSection: renderVideoAnalysisSection,
    renderVideoExplainability: renderVideoExplainability,
    renderVideoAttentionMap: renderVideoAttentionMap,
    bindTamperingMap: bindTamperingMap,
    evidenceReadiness: evidenceReadiness,
  };
})(typeof window !== "undefined" ? window : globalThis);
