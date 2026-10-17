"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const root = path.join(__dirname, "..", "DIGITALEVIDENCE_FIXED");

test("Compare Integrity client sends only the two evidence IDs with session credentials", async () => {
  let request;
  const context = {
    fetch: async (url, options) => {
      request = { url, options };
      return {
        status: 200,
        ok: true,
        headers: { get: () => "application/json" },
        json: async () => ({ ok: true, data: { outcome: "MATCH" } }),
      };
    },
  };
  context.window = context;
  context.globalThis = context;
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.join(root, "api.js"), "utf8"), context);
  const result = await context.MayaApi.evidence.compareIntegrity([17, 23]);
  assert.equal(result.outcome, "MATCH");
  assert.equal(request.url, "/api/evidence/compare-integrity");
  assert.equal(request.options.method, "POST");
  assert.equal(request.options.credentials, "include");
  assert.deepEqual(JSON.parse(request.options.body), { evidence_ids: [17, 23] });
});

test("Compare Integrity view includes selection, outcomes, errors, and case reset behavior", () => {
  const source = fs.readFileSync(path.join(root, "script.js"), "utf8");
  assert.match(source, /'#\/compare-integrity': \{ render: \(\) => renderIntegrityCompareRoute\(\)/);
  assert.match(source, /data-integrity-compare-evidence/);
  assert.match(source, /button\.disabled = rows\.length !== 2/);
  assert.match(source, /if \(boxes\.filter\(item => item\.checked\)\.length > 2\)/);
  assert.match(source, /location\.hash = `#\/compare-integrity\?case=/);
  assert.match(source, /Exact hash match/);
  assert.match(source, /Hash mismatch/);
  assert.match(source, /Unable to compare/);
  assert.match(source, /role="alert"/);
  assert.match(source, /role="status"/);
  assert.match(source, /does not by itself prove malicious tampering/);
  assert.match(source, /matching recorded hash does not verify the current file/);
});
