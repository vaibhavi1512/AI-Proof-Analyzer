"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const root = path.join(__dirname, "..", "DIGITALEVIDENCE_FIXED");
const context = { console };
context.window = context;
context.globalThis = context;
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(root, "api.js"), "utf8"), context);
vm.runInContext(fs.readFileSync(path.join(root, "video_result.js"), "utf8"), context);

const { MayaApi, MayaVideoResult } = context;

function frames(count, options) {
  const opts = options || {};
  const rows = [];
  for (let index = 0; index < count; index += 1) {
    const fallback = opts.fallbackIndexes ? opts.fallbackIndexes.includes(index) : false;
    rows.push({
      frame_index: index * 10,
      timestamp_seconds: index * 0.5,
      face_detected: !fallback,
      used_face_crop: !fallback,
      used_full_frame_fallback: fallback,
      temporal_importance: opts.importance === false ? null : (index + 1) / ((count * (count + 1)) / 2),
    });
  }
  return rows;
}

function rawVideo(overrides) {
  const baseFrames = frames(16, overrides);
  return {
    prediction: overrides.prediction,
    p_fake: overrides.pFake,
    threshold: 0.5,
    frames_analyzed: 16,
    face_crop_frames: overrides.faceCrops,
    fallback_frames: overrides.fallback,
    xai_available: overrides.xai !== false,
    model_name: "ffpp_video_lstm_v2",
    model_version: "v2-16frame",
    p_fake_meaning: "model-predicted probability for the FAKE class",
    frames: baseFrames,
    xai: overrides.xai === false ? null : {
      spatial_wording: "regions contributing to the model prediction",
      temporal_wording: "frames with higher relative contribution",
      frames: baseFrames,
    },
    gradcam_contact_sheet: overrides.contact === true,
    gradcam_frame_indices: overrides.indices || [],
  };
}

function render(raw, analysisId) {
  const analysis = MayaApi.adapt.analysis({
    analysis_id: analysisId || 7,
    prediction: raw.prediction,
    confidence: null,
    analysis_status: "COMPLETED",
    video_analysis: raw,
  });
  return MayaVideoResult.renderVideoAnalysisSection(analysis, {
    analysisId: analysis.analysisId,
    artifactUrl: (id, kind, frame) => `/api/analysis/${id}/artifact/${kind}${frame == null ? "" : "?frame=" + frame}`,
    evidenceLabel: "clip.mp4",
    sha256: "abc123",
  });
}

const artifactUrl = (id, kind, frame) => `/api/analysis/${id}/artifact/${kind}${frame == null ? "" : "?frame=" + frame}`;

function screen(kind, raw, analysisId) {
  const analysis = MayaApi.adapt.analysis({
    analysis_id: analysisId || 7,
    prediction: raw.prediction,
    confidence: null,
    analysis_status: "COMPLETED",
    model_name: raw.model_name,
    model_version: raw.model_version,
    video_analysis: raw,
  });
  const options = {
    analysisId: analysis.analysisId,
    artifactUrl,
    evidenceLabel: "clip.mp4",
    sha256: "abc123",
  };
  if (kind === "xai") return MayaVideoResult.renderVideoExplainability(analysis, options);
  if (kind === "attention") return MayaVideoResult.renderVideoAttentionMap(analysis, options);
  return MayaVideoResult.renderVideoAnalysisSection(analysis, options);
}

test("image analysis without video_analysis does not render the video section", () => {
  const image = MayaApi.adapt.analysis({
    analysis_id: 1,
    prediction: "REAL",
    confidence: 91,
    analysis_status: "COMPLETED",
    model_name: "efficientnet_b0",
  });
  assert.equal(image.videoAnalysis, null);
  assert.equal(image.confidence, 91);
  assert.equal(MayaVideoResult.renderVideoAnalysisSection(image), "");
  assert.equal(MayaVideoResult.renderVideoExplainability(image), "");
  assert.equal(MayaVideoResult.renderVideoAttentionMap(image), "");
  const script = fs.readFileSync(path.join(root, "script.js"), "utf8");
  assert.match(script, /MODEL CONFIDENCE/);
  assert.match(script, /Model confidence:/);
  assert.match(script, /a\.videoAnalysis && window\.MayaVideoResult/);
});

test("FAKE with p_fake 0.7099 shows confidence 70.99 percent", () => {
  const html = render(rawVideo({
    prediction: "FAKE",
    pFake: 0.7099,
    faceCrops: 16,
    fallback: 0,
  }));
  assert.match(html, />FAKE</);
  assert.match(html, /Confidence Score/);
  assert.match(html, /data-field="confidence"[^>]*>70\.99%</);
  assert.doesNotMatch(html, /Model-predicted FAKE probability|legal certainty|guarantee|proof of manipulation/i);
});

test("REAL with p_fake 0.1040 shows confidence 89.60 percent", () => {
  const html = render(rawVideo({
    prediction: "REAL",
    pFake: 0.104,
    faceCrops: 15,
    fallback: 1,
    fallbackIndexes: [3],
  }));
  assert.match(html, />REAL</);
  assert.match(html, /data-field="confidence"[^>]*>89\.60%</);
  assert.doesNotMatch(html, /10\.40%/);
});

test("REAL with p_fake 0.1500 shows confidence 85.00 percent", () => {
  const html = render(rawVideo({
    prediction: "REAL",
    pFake: 0.15,
    faceCrops: 16,
    fallback: 0,
  }));
  assert.match(html, />REAL</);
  assert.match(html, /data-field="confidence"[^>]*>85\.00%</);
  assert.match(html, /data-field="frames">16</);
  assert.match(html, /data-field="face-crops">16</);
  assert.match(html, /data-field="fallback">0</);
  assert.doesNotMatch(html, /15\.00%/);
  assert.doesNotMatch(html, /MODEL CONFIDENCE|Certainty|Authenticity percentage|Proof percentage/);
  assert.doesNotMatch(html, /no detectable face/);
});

test("temporal graph is shown and the frame contribution list is not", () => {
  const html = render(rawVideo({
    prediction: "FAKE",
    pFake: 0.7099,
    faceCrops: 16,
    fallback: 0,
    indices: [0, 10],
    contact: true,
  }));
  assert.match(html, /Temporal Analysis/);
  assert.match(html, /Relative temporal contribution/);
  assert.match(html, /videoTemporalChart/);
  assert.equal(MayaVideoResult.temporalPoints(MayaApi.adapt.analysis({
    analysis_id: 7,
    prediction: "FAKE",
    analysis_status: "COMPLETED",
    video_analysis: rawVideo({ prediction: "FAKE", pFake: 0.7099, faceCrops: 16, fallback: 0 }),
  }).videoAnalysis).length, 16);
  assert.doesNotMatch(html, /video-temporal-list/);
  assert.doesNotMatch(html, /relative contribution \d/);
  assert.doesNotMatch(html, /Frame \d+ —/);
});

test("Grad-CAM displays on the tampering map when the artifact is available", () => {
  const raw = rawVideo({
    prediction: "FAKE",
    pFake: 0.7099,
    faceCrops: 16,
    fallback: 0,
    indices: [0, 10],
    contact: true,
  });
  const analysis = render(raw);
  assert.match(analysis, /data-video-screen="analysis"/);
  assert.doesNotMatch(analysis, /video_gradcam|video_contact_sheet|data-video-frame/);
  const html = screen("attention", raw);
  assert.match(html, /class="video-gradcam"/);
  assert.match(html, /video_gradcam\?frame=0/);
  assert.match(html, /video_gradcam\?frame=10/);
  assert.match(html, /video_contact_sheet/);
  assert.match(html, /Regions contributing to the model prediction/);
  assert.match(html, /face crop/);
  assert.doesNotMatch(html, /Grad-CAM artifact is not available for this analysis/);
});

test("Grad-CAM stays unavailable when the artifact is missing", () => {
  const unavailable = render(rawVideo({
    prediction: "REAL",
    pFake: 0.15,
    faceCrops: 16,
    fallback: 0,
    xai: false,
  }));
  assert.match(unavailable, />REAL</);
  assert.match(unavailable, /data-field="confidence"[^>]*>85\.00%</);
  assert.match(unavailable, /XAI unavailable/);
  assert.doesNotMatch(unavailable, /videoTemporalChart|<img|video_gradcam|video_contact_sheet/);

  const missingRaw = rawVideo({
    prediction: "FAKE",
    pFake: 0.7099,
    faceCrops: 16,
    fallback: 0,
    indices: [],
    contact: false,
  });
  const missingAnalysis = render(missingRaw);
  assert.match(missingAnalysis, />FAKE</);
  assert.match(missingAnalysis, /70\.99%/);
  assert.match(missingAnalysis, /videoTemporalChart/);
  assert.doesNotMatch(missingAnalysis, /<img|video_gradcam|video_contact_sheet/);
  const missingArtifact = screen("attention", missingRaw);
  assert.match(missingArtifact, /Grad-CAM artifact is not available for this analysis/);
  assert.doesNotMatch(missingArtifact, /<img|video_gradcam|video_contact_sheet|videoTemporalChart/);
});

test("no-face fallback wording does not call the frame FAKE", () => {
  const raw = rawVideo({
    prediction: "REAL",
    pFake: 0.104,
    faceCrops: 15,
    fallback: 1,
    fallbackIndexes: [3],
    indices: [30],
  });
  const html = render(raw);
  assert.match(html, /data-field="frames">16</);
  assert.match(html, /data-field="face-crops">15</);
  assert.match(html, /data-field="fallback">1</);
  assert.match(html, /1 sampled frame had no detectable face and was analyzed using the full-frame fallback/);
  assert.match(html, /full-frame fallback/);
  assert.doesNotMatch(html, /NO FACE|fake frame|no-face frame as fake/i);
  assert.doesNotMatch(html, />FAKE</);
  const map = screen("attention", raw);
  assert.match(map, /Frame 30/);
  assert.match(map, /full-frame fallback/);
  assert.doesNotMatch(map, />FAKE</);
});

test("missing numeric fields stay blank", () => {
  const html = render({
    prediction: "REAL",
    xai_available: false,
    frames: [],
  });
  assert.match(html, /data-field="confidence"[^>]*>—</);
  assert.match(html, /data-field="frames">—</);
  assert.match(html, /data-field="face-crops">—</);
  assert.match(html, /data-field="fallback">—</);
});

test("starting a new analysis clears stale video results", () => {
  const first = render(rawVideo({
    prediction: "FAKE",
    pFake: 0.7099,
    faceCrops: 16,
    fallback: 0,
    indices: [0],
    contact: true,
  }), 1);
  assert.match(first, /70\.99%/);
  assert.match(first, /videoTemporalChart/);
  assert.doesNotMatch(first, /video_gradcam\?frame=0/);
  assert.match(screen("attention", rawVideo({
    prediction: "FAKE",
    pFake: 0.7099,
    faceCrops: 16,
    fallback: 0,
    indices: [0],
    contact: true,
  }), 1), /video_gradcam\?frame=0/);

  const pending = MayaVideoResult.beginNewAnalysis();
  assert.match(pending, /data-video-state="pending"/);
  assert.match(pending, /data-analysis-id=""/);
  assert.doesNotMatch(pending, /FAKE|REAL|70\.99%|89\.60%|85\.00%|videoTemporalChart|video_gradcam|video_contact_sheet|<img/);

  const second = render(rawVideo({
    prediction: "REAL",
    pFake: 0.15,
    faceCrops: 16,
    fallback: 0,
    indices: [],
    contact: false,
  }), 2);
  assert.match(second, /data-analysis-id="2"/);
  assert.match(second, />REAL</);
  assert.match(second, /85\.00%/);
  assert.doesNotMatch(second, />FAKE</);
  assert.doesNotMatch(second, /70\.99%/);
  assert.doesNotMatch(second, /video_gradcam|video_contact_sheet/);

  const failed = MayaVideoResult.renderVideoFailure("checkpoint missing");
  assert.match(failed, /data-video-state="failed"/);
  assert.doesNotMatch(failed, />FAKE<|>REAL</);
  const script = fs.readFileSync(path.join(root, "script.js"), "utf8");
  assert.match(script, /MayaVideoResult\.beginNewAnalysis\(\)/);
  assert.match(script, /__videoTemporalChart\.destroy\(\)/);
});

test("attention heat map uses video Grad-CAM routes and omits broken artifact URLs", () => {
  const available = screen("attention", rawVideo({
    prediction: "FAKE",
    pFake: 0.7099,
    faceCrops: 16,
    fallback: 0,
    indices: [0, 12],
    contact: true,
  }));
  assert.match(available, /data-video-screen="attention"/);
  assert.match(available, /Confidence Score/);
  assert.match(available, /70\.99%/);
  assert.match(available, /ffpp_video_lstm_v2/);
  assert.match(available, /v2-16frame/);
  assert.match(available, /video_gradcam\?frame=0/);
  assert.match(available, /video_gradcam\?frame=12/);
  assert.match(available, /video_contact_sheet/);
  assert.doesNotMatch(available, /artifact\/heatmap|artifact\/overlay|artifact\/original/);
  assert.doesNotMatch(available, /Image could not be loaded|Frame confidence|Flagged Regions|Video confidence/);

  const missing = screen("attention", rawVideo({
    prediction: "REAL",
    pFake: 0.104,
    faceCrops: 15,
    fallback: 1,
    indices: [],
    contact: false,
  }));
  assert.match(missing, /89\.60%/);
  assert.match(missing, /Grad-CAM artifact is not available for this analysis/);
  assert.doesNotMatch(missing, /<img|artifact\/heatmap|artifact\/overlay|artifact\/original|Image could not be loaded/);
});

test("explainable AI screen uses the current video analysis values", () => {
  const html = screen("xai", rawVideo({
    prediction: "REAL",
    pFake: 0.104,
    faceCrops: 15,
    fallback: 1,
    fallbackIndexes: [3],
    indices: [30],
    contact: true,
  }));
  assert.match(html, /data-video-screen="xai"/);
  assert.match(html, />REAL</);
  assert.match(html, /Confidence Score/);
  assert.match(html, /89\.60%/);
  assert.match(html, /ffpp_video_lstm_v2/);
  assert.match(html, /v2-16frame/);
  assert.match(html, /Relative temporal contribution/);
  assert.match(html, /videoTemporalChart/);
  assert.match(html, /Regions contributing to the model prediction/);
  assert.match(html, /video_gradcam\?frame=30/);
  assert.match(html, /video_contact_sheet/);
  assert.doesNotMatch(html, /Model confidence: not reported|artifact\/heatmap|artifact\/overlay|proof of manipulation/);
  const script = fs.readFileSync(path.join(root, "script.js"), "utf8");
  assert.match(script, /renderVideoExplainability/);
  assert.match(script, /renderVideoAttentionMap/);
});

function readinessRun(video, extras) {
  return Object.assign({
    prediction: video.prediction,
    confidence: null,
    model_name: video.model_name || "ffpp_video_lstm_v2",
    model_version: video.model_version || "v2-16frame",
    explanation: { explainer: "video_gradcam", heatmap: null, overlay: null },
    video_analysis: video,
  }, extras || {});
}

test("video readiness recognizes FAKE confidence and Grad-CAM", () => {
  const ready = MayaVideoResult.evidenceReadiness(readinessRun({
    prediction: "FAKE",
    p_fake: 0.7099,
    xai_available: true,
    gradcam_contact_sheet: true,
    gradcam_frame_indices: [0, 12],
    model_name: "ffpp_video_lstm_v2",
    model_version: "v2-16frame",
  }, { confidence: 0 }));
  assert.match(ready.analysisNote, /FAKE at 70\.99% confidence \(ffpp_video_lstm_v2 v2-16frame\)/);
  assert.doesNotMatch(ready.analysisNote, /0\.0%/);
  assert.equal(ready.explainabilityOk, true);
  assert.match(ready.explainabilityNote, /Grad-CAM available/);
});

test("video readiness recognizes REAL confidence when XAI is available", () => {
  const ready = MayaVideoResult.evidenceReadiness(readinessRun({
    prediction: "REAL",
    p_fake: 0.104,
    xai_available: true,
    gradcam_contact_sheet: false,
    gradcam_frame_indices: [],
    xai: {
      frames: [{ frame_index: 0, timestamp_seconds: 0, temporal_importance: 1 }],
    },
    model_name: "ffpp_video_lstm_v2",
    model_version: "v2-16frame",
  }));
  assert.match(ready.analysisNote, /REAL at 89\.60% confidence/);
  assert.doesNotMatch(ready.analysisNote, /10\.40%/);
  assert.equal(ready.explainabilityOk, true);
});

test("video readiness leaves explainability unavailable when XAI is missing", () => {
  const missing = MayaVideoResult.evidenceReadiness(readinessRun({
    prediction: "REAL",
    p_fake: 0.15,
    xai_available: false,
    gradcam_contact_sheet: false,
    gradcam_frame_indices: [],
  }));
  assert.match(missing.analysisNote, /REAL at 85\.00% confidence/);
  assert.equal(missing.explainabilityOk, false);
  assert.equal(missing.explainabilityNote, "XAI unavailable");

  const flagOnly = MayaVideoResult.evidenceReadiness(readinessRun({
    prediction: "FAKE",
    p_fake: 0.7099,
    xai_available: true,
    gradcam_contact_sheet: false,
    gradcam_frame_indices: [],
  }));
  assert.equal(flagOnly.explainabilityOk, false);
  assert.match(flagOnly.analysisNote, /70\.99%/);
});

test("image readiness does not use the video readiness result", () => {
  const image = {
    prediction: "REAL",
    confidence: 91,
    model_name: "efficientnet_b0",
    model_version: "v1",
    explanation: { explainer: "gradcam", heatmap: "artifacts/heat.png", overlay: "artifacts/over.png" },
  };
  assert.equal(MayaVideoResult.evidenceReadiness(image), null);
  assert.equal(MayaVideoResult.evidenceReadiness({ prediction: "FAKE", confidence: 80, video_analysis: null }), null);
  const script = fs.readFileSync(path.join(root, "script.js"), "utf8");
  assert.match(script, /Number\(completed\.confidence\)\.toFixed\(1\)/);
  assert.match(script, /completed\.explanation\.heatmap \|\| completed\.explanation\.overlay/);
  assert.match(script, /completed\.explanation\.heatmap \? `Grad-CAM \(\$\{completed\.explanation\.explainer\}\)`/);
  assert.match(script, /completed && completed\.video_analysis && window\.MayaVideoResult/);
});

test("video result includes a generate-report control wired to the existing handler", () => {
  const html = render(rawVideo({
    prediction: "FAKE",
    pFake: 0.62,
    faceCrops: 16,
    fallback: 0,
  }), 44);
  assert.match(html, /data-action="generate-report"/);
  assert.match(html, /data-analysis="44"/);
  assert.match(html, /Generate Report/);
  const script = fs.readFileSync(path.join(root, "script.js"), "utf8");
  assert.match(script, /if \(el\.disabled\) return;/);
  assert.match(script, /el\.disabled = true/);
  assert.match(script, /Generating…/);
  assert.match(script, /Report generated/);
  assert.match(script, /Report generation failed/);
});

test("video analysis, xai insights, and tampering map are different views", () => {
  const raw = rawVideo({
    prediction: "FAKE",
    pFake: 0.7099,
    faceCrops: 14,
    fallback: 2,
    fallbackIndexes: [1],
    indices: [0, 10],
    contact: true,
  });
  const analysis = screen("analysis", raw);
  const xai = screen("xai", raw);
  const map = screen("attention", raw);
  assert.notEqual(analysis, xai);
  assert.notEqual(xai, map);
  assert.notEqual(analysis, map);

  assert.match(analysis, /data-video-screen="analysis"/);
  assert.match(analysis, /data-field="frames">16</);
  assert.match(analysis, /data-field="face-crops">14</);
  assert.match(analysis, /data-field="fallback">2</);
  assert.match(analysis, /videoTemporalChart/);
  assert.match(analysis, /v2-16frame/);
  assert.doesNotMatch(analysis, /video_gradcam|video_contact_sheet|data-video-frame|Tampering Map|Spatial explanation summary/);

  assert.match(xai, /data-video-screen="xai"/);
  assert.match(xai, /XAI Insights/);
  assert.match(xai, /Relative temporal contribution/);
  assert.match(xai, /videoTemporalChart/);
  assert.match(xai, /Spatial explanation summary/);
  assert.match(xai, /Regions contributing to the model prediction/);
  assert.match(xai, /video_contact_sheet/);
  assert.match(xai, /video_gradcam\?frame=0/);
  assert.doesNotMatch(xai, /video_gradcam\?frame=10|data-video-frame|data-field="frames"/);
  assert.doesNotMatch(xai, /proof of manipulation/);

  assert.match(map, /data-video-screen="attention"/);
  assert.match(map, /Tampering Map/);
  assert.match(map, /video_contact_sheet/);
  assert.match(map, /video_gradcam\?frame=0/);
  assert.match(map, /data-video-frame="10"/);
  assert.match(map, /Frame 0 · 0\.00s/);
  assert.match(map, /Frame 10/);
  assert.doesNotMatch(map, /videoTemporalChart|artifact\/heatmap|artifact\/overlay|artifact\/original|Spatial explanation summary/);

  const missing = rawVideo({
    prediction: "REAL",
    pFake: 0.15,
    faceCrops: 16,
    fallback: 0,
    xai: false,
  });
  const missingXai = screen("xai", missing);
  const missingMap = screen("attention", missing);
  assert.match(screen("analysis", missing), />REAL</);
  assert.match(screen("analysis", missing), /85\.00%/);
  assert.match(missingXai, /XAI unavailable/);
  assert.match(missingMap, /XAI unavailable/);
  assert.doesNotMatch(missingXai, /<img|videoTemporalChart|Spatial explanation summary/);
  assert.doesNotMatch(missingMap, /<img|videoTemporalChart|video_gradcam|video_contact_sheet/);

  const script = fs.readFileSync(path.join(root, "script.js"), "utf8");
  assert.match(script, /function withCurrentEvidence/);
  assert.match(script, /'#\/analysis': true/);
  assert.match(script, /'#\/xai': true/);
  assert.match(script, /'#\/tampering': true/);
  assert.match(script, /MayaVideoResult\.bindTamperingMap\(document\)/);
  assert.match(script, /renderVideoAnalysisSection/);
  assert.match(script, /renderVideoExplainability/);
  assert.match(script, /renderVideoAttentionMap/);
  assert.match(script, /artifactUrl\(a\.analysisId, 'overlay'/);
  assert.match(script, /artifactUrl\(a\.analysisId, 'heatmap'/);
  assert.match(script, /MODEL CONFIDENCE/);
});

test("empty image heatmap fields do not hide an available video Grad-CAM", () => {
  const ready = MayaVideoResult.evidenceReadiness(readinessRun({
    prediction: "FAKE",
    p_fake: 0.7099,
    xai_available: true,
    gradcam_contact_sheet: true,
    gradcam_frame_indices: [4],
  }));
  assert.equal(ready.explainabilityOk, true);
  assert.match(ready.analysisNote, /70\.99%/);
  assert.equal(ready.explainabilityNote, "Grad-CAM available for this video analysis");
});
