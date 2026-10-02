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
vm.runInContext(fs.readFileSync(path.join(root, "password_policy.js"), "utf8"), context);

const { MayaPassword } = context;
const script = fs.readFileSync(path.join(root, "script.js"), "utf8");
const page = fs.readFileSync(path.join(root, "index.html"), "utf8");
const evidex = fs.readFileSync(path.join(__dirname, "..", "backend", "app", "routes", "evidex.py"), "utf8");

function item(rule, label) {
  const classes = new Set();
  return {
    dataset: { rule, label },
    textContent: "☐ " + label,
    classList: {
      toggle(name, on) {
        if (on) classes.add(name);
        else classes.delete(name);
      },
      has(name) { return classes.has(name); },
    },
  };
}

function form(password) {
  const button = { disabled: true };
  const items = [
    item("length", "At least 8 characters"),
    item("uppercase", "1 uppercase letter"),
    item("lowercase", "1 lowercase letter"),
    item("number", "1 number"),
    item("special", "1 special character"),
  ];
  const result = MayaPassword.applyChecklist(password, items, button);
  return { result, items, button };
}

function rule(items, name) {
  return items.find((entry) => entry.dataset.rule === name);
}

test("empty password leaves every requirement unchecked and registration disabled", () => {
  const view = form("");
  assert.equal(MayaPassword.satisfied(view.result), false);
  for (const entry of view.items) {
    assert.equal(view.result[entry.dataset.rule], false);
    assert.equal(entry.classList.has("met"), false);
    assert.match(entry.textContent, /^☐ /);
  }
  assert.equal(view.button.disabled, true);
  assert.equal(MayaPassword.registrationReady({
    password: "",
    username: "investigator",
    email: "investigator@maya.test",
  }), false);
});

test("a short password keeps registration disabled", () => {
  const view = form("Ab@1");
  assert.equal(view.result.length, false);
  assert.equal(rule(view.items, "length").classList.has("met"), false);
  assert.equal(view.button.disabled, true);
});

test("missing uppercase stays unchecked", () => {
  const view = form("abcd@123");
  assert.equal(view.result.uppercase, false);
  assert.equal(rule(view.items, "uppercase").classList.has("met"), false);
  assert.match(rule(view.items, "uppercase").textContent, /^☐ /);
  assert.equal(view.button.disabled, true);
});

test("missing lowercase stays unchecked", () => {
  const view = form("ABCD@123");
  assert.equal(view.result.lowercase, false);
  assert.equal(rule(view.items, "lowercase").classList.has("met"), false);
  assert.equal(view.button.disabled, true);
});

test("missing number stays unchecked", () => {
  const view = form("Abcdefg@");
  assert.equal(view.result.number, false);
  assert.equal(rule(view.items, "number").classList.has("met"), false);
  assert.equal(view.button.disabled, true);
});

test("missing special character stays unchecked", () => {
  const view = form("Abcdefg1");
  assert.equal(view.result.special, false);
  assert.equal(rule(view.items, "special").classList.has("met"), false);
  assert.equal(view.button.disabled, true);
});

test("Abcd@123 checks every rule and can enable registration", () => {
  const view = form("Abcd@123");
  assert.equal(view.result.length, true);
  assert.equal(view.result.uppercase, true);
  assert.equal(view.result.lowercase, true);
  assert.equal(view.result.number, true);
  assert.equal(view.result.special, true);
  for (const entry of view.items) assert.match(entry.textContent, /^✓ /);
  assert.equal(view.button.disabled, false);
  assert.equal(MayaPassword.registrationReady({
    password: "Abcd@123",
    username: "investigator",
    email: "investigator@maya.test",
  }), true);
  assert.equal(MayaPassword.registrationReady({
    password: "Abcd@123",
    username: "",
    email: "investigator@maya.test",
  }), false);
});

test("a longer valid password with @ checks every rule", () => {
  const view = form("Vedanti@1204");
  assert.equal(view.result.length, true);
  assert.equal(view.result.uppercase, true);
  assert.equal(view.result.lowercase, true);
  assert.equal(view.result.number, true);
  assert.equal(view.result.special, true);
  assert.equal(view.button.disabled, false);
  const removed = form("Vedanti1204");
  assert.equal(removed.result.special, false);
  assert.equal(removed.button.disabled, true);
  assert.equal(MayaPassword.registrationReady({
    password: "Vedanti@1204",
    username: "vedanti",
    email: "vedanti@maya.test",
  }), true);
});

test("script.js still validates when the separate policy file is missing", () => {
  assert.match(script, /if \(!window\.MayaPassword\)/);
  assert.match(script, /id="regPassword"/);
  assert.match(script, /addEventListener\('input', refreshPasswordRequirements\)/);
});

test("editing a valid password back to invalid disables registration", () => {
  const button = { disabled: true };
  const items = [item("special", "1 special character")];
  MayaPassword.applyChecklist("Abcd@123", items, button);
  assert.equal(button.disabled, false);
  assert.match(items[0].textContent, /^✓ /);
  MayaPassword.applyChecklist("Abcd1234", items, button);
  assert.equal(button.disabled, true);
  assert.match(items[0].textContent, /^☐ /);
  assert.equal(items[0].classList.has("met"), false);
});

test("show/hide does not change the password result", () => {
  const hidden = MayaPassword.checks("Abcd@123");
  const shown = MayaPassword.checks("Abcd@123");
  assert.equal(MayaPassword.satisfied(hidden), true);
  assert.equal(MayaPassword.satisfied(shown), true);
  assert.match(script, /input\.type = visible \? 'password' : 'text'/);
  assert.match(script, /if \(input === regPassword\) refreshPasswordRequirements\(\)/);
});

test("registration still posts the same fields and the policy script is served", () => {
  assert.match(script, /type="password" id="regPassword"|id="regPassword"[^>]*type="password"/);
  assert.match(script, /MayaPassword\.registrationReady/);
  assert.match(script, /addEventListener\('input', refreshPasswordRequirements\)/);
  assert.match(script, /full_name:/);
  assert.match(script, /username:/);
  assert.match(script, /email:/);
  assert.match(script, /password:/);
  assert.match(page, /password_policy\.js/);
  assert.match(evidex, /password_policy\.js/);
});
