/* Shared password rules for registration. The backend enforces the same rules. */
(function (global) {
  'use strict';

  function checks(password) {
    const value = String(password || '');
    return {
      length: value.length >= 8,
      uppercase: /[A-Z]/.test(value),
      lowercase: /[a-z]/.test(value),
      number: /[0-9]/.test(value),
      special: /[^A-Za-z0-9]/.test(value),
    };
  }

  function satisfied(result) {
    return !!(result && result.length && result.uppercase && result.lowercase && result.number && result.special);
  }

  function applyChecklist(password, items, button) {
    const result = checks(password);
    (items || []).forEach(function (item) {
      const ok = !!result[item.dataset.rule];
      item.classList.toggle('met', ok);
      const label = item.dataset.label || '';
      item.textContent = (ok ? '✓ ' : '☐ ') + label;
    });
    if (button) button.disabled = !satisfied(result);
    return result;
  }

  function registrationReady(fields) {
    const source = fields || {};
    const username = String(source.username || '').trim();
    const email = String(source.email || '').trim();
    const passwordOk = satisfied(checks(source.password));
    const emailOk = /^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email);
    return passwordOk && username.length >= 3 && emailOk;
  }

  global.MayaPassword = {
    checks: checks,
    satisfied: satisfied,
    applyChecklist: applyChecklist,
    registrationReady: registrationReady,
  };
})(typeof window !== 'undefined' ? window : globalThis);
