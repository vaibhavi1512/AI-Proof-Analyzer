const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const root = path.resolve(__dirname, "..", "DIGITALEVIDENCE_FIXED");
const script = fs.readFileSync(path.join(root, "script.js"), "utf8");
const api = fs.readFileSync(path.join(root, "api.js"), "utf8");
const video = fs.readFileSync(path.join(root, "video_result.js"), "utf8");
const frontend = script + "\n" + api + "\n" + video;

test("report generation no longer asks for a recipient", () => {
  assert.doesNotMatch(frontend, /Send Report by Email|reportEmailTo|email-report|emailReport/);
  assert.doesNotMatch(frontend, /MAIL_PASSWORD|MAIL_USERNAME|MAIL_HOST|smtp/i);
  assert.match(script, /Generate Report/);
  assert.match(script, /Download PDF/);
  assert.match(script, /emailMessage/);
  assert.match(script, /maskedRecipient/);
  assert.match(api, /emailStatus: r\.email_delivery \? r\.email_delivery\.status/);
  assert.match(video, /Generate Report/);
  assert.doesNotMatch(api, /\/report\/email/);
});
