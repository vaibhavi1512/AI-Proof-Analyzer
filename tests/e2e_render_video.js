"use strict";

const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const input = JSON.parse(fs.readFileSync(0, "utf8"));
const root = path.join(__dirname, "..", "DIGITALEVIDENCE_FIXED");
const context = { console };
context.window = context;
context.globalThis = context;
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(root, "api.js"), "utf8"), context);
vm.runInContext(fs.readFileSync(path.join(root, "video_result.js"), "utf8"), context);

const analysis = context.MayaApi.adapt.analysis(input);
if (!analysis.videoAnalysis) {
  process.stdout.write("");
  process.exit(0);
}
const options = {
  analysisId: analysis.analysisId,
  artifactUrl: (id, kind, frame) => (
    `/api/analysis/${id}/artifact/${kind}` + (frame == null ? "" : `?frame=${frame}`)
  ),
  evidenceLabel: input.original_filename || "video.mp4",
  sha256: input._sha256 || "",
};
const screen = input._screen || "analysis";
const render = screen === "xai"
  ? context.MayaVideoResult.renderVideoExplainability
  : screen === "attention"
    ? context.MayaVideoResult.renderVideoAttentionMap
    : context.MayaVideoResult.renderVideoAnalysisSection;
process.stdout.write(render(analysis, options));
