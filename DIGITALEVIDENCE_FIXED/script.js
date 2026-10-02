/* ==========================================================================
   EVIDEX — Intelligent Digital Evidence Verification
   Frontend application logic: routing, auth, mock forensic simulation,
   charts, chain-of-custody, XAI, heatmap, admin & reporting views.

   This is a fully client-side demo. All "AI analysis" is simulated with
   seeded pseudo-random data — there is no real model behind it. All
   security shown (chain-of-custody password, hashing) is a frontend demo
   and must be re-implemented server-side for real legal use.
   ========================================================================== */

/* Registration rules. password_policy.js sets the same object when that file
   loads. This copy runs when that request is missing, so the checklist still
   updates. */
if (!window.MayaPassword) {
  window.MayaPassword = (function () {
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
        item.textContent = (ok ? '✓ ' : '☐ ') + (item.dataset.label || '');
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
    return { checks: checks, satisfied: satisfied, applyChecklist: applyChecklist, registrationReady: registrationReady };
  })();
}

/* ==========================================================================
   1. ICONS  (inline SVG, stroke-based, consistent 24x24 viewbox)
   ========================================================================== */
const ICONS = {
  logo: '<svg viewBox="0 0 24 24" fill="none"><path d="M12 2L3 6v6c0 5 4 8.5 9 10 5-1.5 9-5 9-10V6l-9-4z" fill="currentColor"/><path d="M9 12l2 2 4-4.5" stroke="#04141a" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  dashboard: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><rect x="3" y="3" width="7" height="9" rx="1.5"/><rect x="14" y="3" width="7" height="5" rx="1.5"/><rect x="14" y="12" width="7" height="9" rx="1.5"/><rect x="3" y="16" width="7" height="5" rx="1.5"/></svg>',
  upload: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 16V4M12 4l-4 4M12 4l4 4"/><path d="M4 16v3a2 2 0 002 2h12a2 2 0 002-2v-3"/></svg>',
  cases: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 8a2 2 0 012-2h4l2 2h8a2 2 0 012 2v7a2 2 0 01-2 2H5a2 2 0 01-2-2V8z"/></svg>',
  analysis: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/><path d="M8 11h6M11 8v6"/></svg>',
  xai: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a5 5 0 015 5c0 2-1 3-2 4.2-.6.7-1 1.3-1 2.3v1.5H10v-1.5c0-1-.4-1.6-1-2.3C8 10 7 9 7 7a5 5 0 015-5z"/><path d="M9.5 19h5M10 22h4"/></svg>',
  tampering: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2l9 4v6c0 5-3.8 8.7-9 10-5.2-1.3-9-5-9-10V6l9-4z"/><path d="M12 8v5M12 16h.01"/></svg>',
  custody: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="10" width="16" height="11" rx="2"/><path d="M8 10V7a4 4 0 018 0v3"/></svg>',
  reports: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M7 3h7l5 5v13a1 1 0 01-1 1H7a1 1 0 01-1-1V4a1 1 0 011-1z"/><path d="M14 3v5h5M9 13h6M9 17h6M9 9h2"/></svg>',
  legal: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v18M5 7l-3 6a3.5 3.5 0 007 0l-3-6zM19 7l-3 6a3.5 3.5 0 007 0l-3-6zM5 7h14M9 3h6"/></svg>',
  profile: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="4"/><path d="M4 21c0-4.4 3.6-7 8-7s8 2.6 8 7"/></svg>',
  logout: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4M16 17l5-5-5-5M21 12H9"/></svg>',
  users: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="9" cy="8" r="3.2"/><path d="M2.5 20c0-3.6 3-6 6.5-6s6.5 2.4 6.5 6"/><path d="M16 4.2a3.2 3.2 0 010 6.2M21.5 20c0-3-2-5.3-5-6"/></svg>',
  investigators: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/><path d="M8 12l2 2 4-4.5"/></svg>',
  evidence: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"/></svg>',
  activity: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h4l2-7 4 14 2-7h6"/></svg>',
  logs: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M9 5H6a2 2 0 00-2 2v12a2 2 0 002 2h12a2 2 0 002-2v-4M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2M13 13l7-7 3 3-7 7h-3v-3z"/></svg>',
  security: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2l8 3.5v5.5c0 5-3.4 8.7-8 10-4.6-1.3-8-5-8-10V5.5L12 2z"/><path d="M9 12l2 2 4-4.5"/></svg>',
  settings: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 00.3 1.9l.1.1a2 2 0 11-2.9 2.9l-.1-.1a1.7 1.7 0 00-1.9-.3 1.7 1.7 0 00-1 1.6V21a2 2 0 11-4 0v-.2a1.7 1.7 0 00-1-1.6 1.7 1.7 0 00-1.9.3l-.1.1a2 2 0 11-2.9-2.9l.1-.1a1.7 1.7 0 00.3-1.9 1.7 1.7 0 00-1.6-1H3a2 2 0 110-4h.2a1.7 1.7 0 001.6-1 1.7 1.7 0 00-.3-1.9l-.1-.1a2 2 0 112.9-2.9l.1.1a1.7 1.7 0 001.9.3H9a1.7 1.7 0 001-1.6V3a2 2 0 114 0v.2a1.7 1.7 0 001 1.6 1.7 1.7 0 001.9-.3l.1-.1a2 2 0 112.9 2.9l-.1.1a1.7 1.7 0 00-.3 1.9V9a1.7 1.7 0 001.6 1H21a2 2 0 110 4h-.2a1.7 1.7 0 00-1.6 1z"/></svg>',
  search: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/></svg>',
  download: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v12m0 0l-4-4m4 4l4-4M4 17v2a2 2 0 002 2h12a2 2 0 002-2v-2"/></svg>',
  print: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9V3h12v6M6 18H4a1 1 0 01-1-1v-5a1 1 0 011-1h16a1 1 0 011 1v5a1 1 0 01-1 1h-2M6 14h12v7H6v-7z"/></svg>',
  lock: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="10" width="16" height="11" rx="2"/><path d="M8 10V7a4 4 0 018 0v3"/></svg>',
  check: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"/></svg>',
  checkCircle: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M8 12l2.5 2.5L16 9"/></svg>',
  alert: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 9v4M12 17h.01"/><path d="M10.3 3.9L2.5 17a1.8 1.8 0 001.6 2.7h15.8a1.8 1.8 0 001.6-2.7L13.7 3.9a1.8 1.8 0 00-3.4 0z"/></svg>',
  alertOctagon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M7.9 2h8.2L22 7.9v8.2L16.1 22H7.9L2 16.1V7.9L7.9 2z"/><path d="M12 8v5M12 16h.01"/></svg>',
  xCircle: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M15 9l-6 6M9 9l6 6"/></svg>',
  helpCircle: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M9.5 9a2.5 2.5 0 015 .5c0 1.7-2.5 2-2.5 3.5M12 17h.01"/></svg>',
  image: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg>',
  video: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="5" width="15" height="14" rx="2"/><path d="M17 10l5-3v10l-5-3"/></svg>',
  audio: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18V5l10-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="16" cy="16" r="3"/></svg>',
  doc: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M7 3h7l5 5v13a1 1 0 01-1 1H7a1 1 0 01-1-1V4a1 1 0 011-1z"/><path d="M14 3v5h5M9 13h6M9 17h6"/></svg>',
  chevron: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9l6 6 6-6"/></svg>',
  chevronRight: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 6l6 6-6 6"/></svg>',
  menu: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18M3 12h18M3 18h18"/></svg>',
  close: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6L6 18M6 6l12 12"/></svg>',
  copy: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>',
  arrowRight: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg>',
  globe: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a15 15 0 010 18M12 3a15 15 0 000 18"/></svg>',
  hash: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M5 9h14M5 15h14M10 3L7 21M17 3l-3 18"/></svg>',
  layers: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2l9 5-9 5-9-5 9-5z"/><path d="M3 12l9 5 9-5M3 17l9 5 9-5"/></svg>',
  scan: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7V5a2 2 0 012-2h2M17 3h2a2 2 0 012 2v2M21 17v2a2 2 0 01-2 2h-2M7 21H5a2 2 0 01-2-2v-2M3 12h18"/></svg>',
  eye: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7z"/><circle cx="12" cy="12" r="3"/></svg>',
  clock: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 3"/></svg>',
  gavel: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M14 6l4 4M5 15l5-5 4 4-5 5H5v-4z"/><path d="M17 3l4 4M2 21h9"/></svg>',
  trash: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13"/><path d="M10 11v6M14 11v6"/></svg>',
};
function ic(name, cls) { return `<span class="ic ${cls||''}" aria-hidden="true">${ICONS[name]||''}</span>`; }

/* ==========================================================================
   2. UTILITIES
   ========================================================================== */
const $ = (sel, root) => (root||document).querySelector(sel);
const $$ = (sel, root) => Array.from((root||document).querySelectorAll(sel));
const escapeHtml = (s) => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

function fmtBytes(bytes) {
  if (bytes < 1024) return bytes + ' B';
  const units = ['KB','MB','GB'];
  let u = -1;
  do { bytes /= 1024; u++; } while (bytes >= 1024 && u < units.length - 1);
  return bytes.toFixed(1) + ' ' + units[u];
}
function fmtDate(d) {
  const dt = (d instanceof Date) ? d : new Date(d);
  return dt.toLocaleString('en-US', { year:'numeric', month:'short', day:'2-digit', hour:'2-digit', minute:'2-digit' });
}
function fmtDateShort(d) {
  const dt = (d instanceof Date) ? d : new Date(d);
  return dt.toLocaleDateString('en-US', { year:'numeric', month:'short', day:'2-digit' });
}
function uid(prefix) {
  return prefix + '-' + Math.random().toString(36).slice(2, 6).toUpperCase() + Date.now().toString(36).slice(-4).toUpperCase();
}

/* Seeded PRNG so mock data + "AI analysis" is stable per evidence id */
function seededRandom(seed) {
  let s = 0;
  for (let i = 0; i < seed.length; i++) s = (s * 31 + seed.charCodeAt(i)) >>> 0;
  return function () {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/* Real SHA-256 via Web Crypto API (used for both file hashing simulation and custody password check) */
async function sha256Hex(input) {
  const enc = new TextEncoder();
  const data = typeof input === 'string' ? enc.encode(input) : input;
  const digest = await crypto.subtle.digest('SHA-256', data);
  return Array.from(new Uint8Array(digest)).map(b => b.toString(16).padStart(2, '0')).join('');
}

function downloadTextFile(filename, text) {
  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/* ==========================================================================
   3. TOASTS & MODALS
   ========================================================================== */
function toast(title, msg, type='success', ms=4200) {
  const stack = $('#toastStack');
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  const iconName = type === 'success' ? 'checkCircle' : type === 'error' ? 'xCircle' : 'alert';
  el.innerHTML = `${ic(iconName, 'toast-ic')}<div><div class="tt">${escapeHtml(title)}</div><div class="tm">${escapeHtml(msg||'')}</div></div>`;
  stack.appendChild(el);
  setTimeout(() => { el.style.transition = 'opacity .3s'; el.style.opacity = '0'; setTimeout(() => el.remove(), 300); }, ms);
}

function openModal(html) {
  $('#modalBox').innerHTML = html;
  $('#modalOverlay').classList.add('show');
}
function closeModal() { $('#modalOverlay').classList.remove('show'); }
document.addEventListener('click', (e) => { if (e.target.id === 'modalOverlay') closeModal(); });

function confirmModal(title, body, onConfirm, confirmLabel='Confirm', danger=true) {
  openModal(`
    <h3>${escapeHtml(title)}</h3>
    <p style="color:var(--text-muted);font-size:13.5px;">${body}</p>
    <div class="modal-actions">
      <button class="btn btn-ghost btn-sm" data-close-modal>Cancel</button>
      <button class="btn ${danger?'btn-danger':'btn-primary'} btn-sm" id="modalConfirmBtn">${escapeHtml(confirmLabel)}</button>
    </div>
  `);
  $('#modalConfirmBtn').addEventListener('click', () => { onConfirm(); closeModal(); });
  $$('[data-close-modal]').forEach(b => b.addEventListener('click', closeModal));
}

/* ==========================================================================
   4. MOCK DATABASE  (seeded once into localStorage, then treated as "backend")
   ========================================================================== */
const DB_KEY = 'evidex_db_v1';
const AUTH_KEY = 'evidex_auth_v1';
const CTX_KEY = 'evidex_ctx_v1';

const INVESTIGATOR_NAMES = ['R. Anand Iyer', 'Priya Nataraj', 'D. Mercer', 'S. Okafor', 'L. Fontaine'];
const EVIDENCE_TYPES = ['Image', 'Video', 'Audio', 'Document'];
const CASE_NAMES = [
  'State vs. Kessler — Financial Fraud', 'Property Dispute — Rowan Estate', 'Cyber Harassment Complaint #4471',
  'Insurance Claim Review — Meridian Auto', 'Corporate IP Theft Investigation', 'Domestic Incident Report #219',
  'Digital Forgery — Contract Exhibit B', 'Missing Persons — Media Review', 'Workplace Misconduct Inquiry',
  'Election Complaint — Media Verification'
];

function seedDB() {
  const rand = seededRandom('evidex-seed-2026');
  const pick = (arr) => arr[Math.floor(rand() * arr.length)];

  const cases = [];
  const evidenceList = [];

  for (let i = 0; i < 10; i++) {
    const caseId = 'CASE-' + (2400 + i);
    const investigator = pick(INVESTIGATOR_NAMES);
    const evCount = 2 + Math.floor(rand() * 4);
    const evidenceIds = [];

    for (let j = 0; j < evCount; j++) {
      const evId = 'EVX-' + (10000 + evidenceList.length);
      const type = pick(EVIDENCE_TYPES);
      const score = Math.round((40 + rand() * 58) * 10) / 10;
      let status = 'Authentic';
      if (score < 55) status = 'Tampered';
      else if (score < 70) status = 'Suspicious';
      else if (score < 78) status = 'Inconclusive';
      const risk = score < 55 ? 'High' : score < 75 ? 'Medium' : 'Low';
      const daysAgo = Math.floor(rand() * 60);
      const date = new Date(Date.now() - daysAgo * 86400000 - Math.floor(rand()*80000000));

      evidenceList.push({
        id: evId,
        caseId: caseId,
        filename: pick(['CCTV_clip_0'+j, 'photo_evidence_', 'statement_scan_', 'call_recording_', 'doc_exhibit_']) + (100+j) + (type==='Image'?'.jpg':type==='Video'?'.mp4':type==='Audio'?'.wav':'.pdf'),
        type,
        size: Math.floor(400000 + rand() * 60000000),
        sha256: null,
        timestamp: date.toISOString(),
        score,
        status,
        risk,
        investigator,
      });
      evidenceIds.push(evId);
    }

    const overallRisk = evidenceList.filter(e => e.caseId === caseId).some(e => e.risk === 'High') ? 'High'
      : evidenceList.filter(e => e.caseId === caseId).some(e => e.risk === 'Medium') ? 'Medium' : 'Low';
    const statuses = ['Open', 'Under Review', 'Closed'];

    cases.push({
      id: caseId,
      name: pick(CASE_NAMES) + (rand() > 0.7 ? '' : ''),
      investigator,
      evidenceIds,
      risk: overallRisk,
      status: statuses[Math.floor(rand()*statuses.length)],
      updated: new Date(Date.now() - Math.floor(rand()*20)*86400000).toISOString(),
      opened: new Date(Date.now() - (30 + Math.floor(rand()*120))*86400000).toISOString(),
    });
  }

  const users = [
    { id: 'USR-001', name: 'Public User Demo', username: 'public_demo', role: 'Public', status: 'Active', joined: '2026-02-11' },
    { id: 'USR-002', name: 'R. Anand Iyer', username: 'investigator', role: 'Investigator', status: 'Active', joined: '2025-11-04' },
    { id: 'USR-003', name: 'Priya Nataraj', username: 'p.nataraj', role: 'Investigator', status: 'Active', joined: '2025-12-19' },
    { id: 'USR-004', name: 'D. Mercer', username: 'd.mercer', role: 'Investigator', status: 'Suspended', joined: '2026-01-08' },
    { id: 'USR-005', name: 'S. Okafor', username: 's.okafor', role: 'Investigator', status: 'Active', joined: '2026-03-22' },
    { id: 'USR-006', name: 'System Administrator', username: 'admin', role: 'Admin', status: 'Active', joined: '2025-09-01' },
    { id: 'USR-007', name: 'Guest Verifier', username: 'guest221', role: 'Public', status: 'Active', joined: '2026-04-30' },
    { id: 'USR-008', name: 'L. Fontaine', username: 'l.fontaine', role: 'Investigator', status: 'Active', joined: '2026-05-14' },
  ];

  const logs = [];
  const logActions = [
    ['AUTH', 'Investigator login', 'investigator'], ['UPLOAD', 'Evidence uploaded', 'p.nataraj'],
    ['ANALYSIS', 'Tampering scan completed', 'system'], ['CUSTODY', 'Chain of custody unlocked', 's.okafor'],
    ['REPORT', 'Legal admissibility report generated', 'investigator'], ['AUTH', 'Failed login attempt', 'unknown'],
    ['ADMIN', 'User account suspended', 'admin'], ['ANALYSIS', 'Deepfake detection flagged high risk', 'system'],
    ['AUTH', 'Admin login', 'admin'], ['EXPORT', 'XAI explanation downloaded', 'd.mercer'],
  ];
  for (let i = 0; i < 14; i++) {
    const [type, msg, actor] = pick(logActions);
    logs.push({ id: 'LOG-'+(9000+i), type, message: msg, actor, time: new Date(Date.now() - i*3600000 - Math.floor(rand()*3000000)).toISOString(),
      severity: type === 'AUTH' && msg.includes('Failed') ? 'warning' : type === 'ADMIN' ? 'info' : 'success' });
  }

  return { cases, evidenceList, users, logs };
}

function getDB() {
  let db = JSON.parse(localStorage.getItem(DB_KEY) || 'null');
  if (!db) { db = seedDB(); localStorage.setItem(DB_KEY, JSON.stringify(db)); }
  return db;
}
function saveDB(db) { localStorage.setItem(DB_KEY, JSON.stringify(db)); }

function getCtx() { return JSON.parse(localStorage.getItem(CTX_KEY) || '{}'); }
function setCtx(patch) { localStorage.setItem(CTX_KEY, JSON.stringify({ ...getCtx(), ...patch })); }

/* ==========================================================================
   5. AUTH
   ========================================================================== */
/* The hardcoded demo credentials and the client-side custody password that
   used to live here were removed: authentication is now performed by the MAYA
   backend, so no secret is shipped in this bundle. */

/* Auth is owned by the MAYA backend (Flask-Login session cookie). The cached
   copy below is display-only chrome (sidebar name/initials); the cookie is the
   real credential and every protected call is authorised server-side. */
function getAuth() { return JSON.parse(localStorage.getItem(AUTH_KEY) || 'null'); }
function setAuth(user) { localStorage.setItem(AUTH_KEY, JSON.stringify(user)); }
function clearAuth() { localStorage.removeItem(AUTH_KEY); }

/* Map a backend user record onto the chrome shape the views already read. */
function authFromBackendUser(user) {
  return {
    role: user.role === 'ADMIN' ? 'admin' : 'investigator',
    backendRole: user.role,
    name: user.full_name || user.username,
    username: user.username,
    email: user.email,
    userId: user.id,
  };
}

/* Confirm the session with the backend, refreshing the cached chrome. */
async function syncAuth() {
  try {
    const user = await MayaApi.auth.me();
    const auth = authFromBackendUser(user);
    setAuth(auth);
    return auth;
  } catch (err) {
    if (err instanceof MayaApi.ApiError && err.isAuth) clearAuth();
    return null;
  }
}

function requireAuth(role) {
  const auth = getAuth();
  if (!auth) { location.hash = '#/login'; return null; }
  if (role && auth.role !== role) { location.hash = '#/login'; return null; }
  return auth;
}

/* Shared error surface: shows the real backend status/message and bounces to
   login on 401 instead of pretending the request succeeded. */
function handleApiError(err, context) {
  if (err instanceof MayaApi.ApiError) {
    if (err.isAuth) {
      clearAuth();
      toast('Session expired', err.message, 'error');
      location.hash = '#/login';
      return;
    }
    toast(context || 'Request failed', `${err.message} (${err.status} ${err.code})`, 'error', 6000);
    return;
  }
  console.error(err);
  toast(context || 'Unexpected error', err.message || String(err), 'error', 6000);
}

/* Standard panels for data MAYA genuinely does not produce. Used instead of
   fabricating forensic values. */
function unavailablePanel(title, reason) {
  return `
    <div class="card">
      <div class="panel-title"><h3>${escapeHtml(title)}</h3><span class="sub">Not available</span></div>
      <p style="color:var(--text-muted);font-size:13.5px;margin:0;">${escapeHtml(reason)}</p>
    </div>`;
}

function unavailableInline(reason) {
  return `<span style="color:var(--text-muted);">Not available — ${escapeHtml(reason)}</span>`;
}

function loadingHtml(label) {
  return `<div class="card"><p style="color:var(--text-muted);margin:0;">${escapeHtml(label || 'Loading…')}</p></div>`;
}

/* Demo badge for the pages still backed by the seeded localStorage dataset. */
function demoBanner(what) {
  return `
    <div class="status-banner" style="border-color:var(--amber);background:var(--amber-dim);">
      <div class="sb-icon" style="background:rgba(0,0,0,0.2);color:inherit;">${ic('alert')}</div>
      <div><h3>Demo data</h3><p>${escapeHtml(what)} is not connected to the MAYA backend and shows seeded sample data.</p></div>
    </div>`;
}

function initials(name) {
  return name.split(' ').filter(Boolean).slice(0,2).map(w => w[0]).join('').toUpperCase();
}

/* ==========================================================================
   6. NAV CONFIG PER ROLE
   ========================================================================== */
const NAV_INVESTIGATOR = [
  { group: 'Workspace', items: [
    { path: '#/dashboard', label: 'Dashboard', icon: 'dashboard' },
    { path: '#/upload', label: 'New Evidence', icon: 'upload' },
    { path: '#/cases', label: 'Cases', icon: 'cases' },
  ]},
  { group: 'Forensics', items: [
    { path: '#/analysis', label: 'Analysis', icon: 'analysis' },
    { path: '#/xai', label: 'XAI Insights', icon: 'xai' },
    { path: '#/tampering', label: 'Tampering Map', icon: 'tampering' },
    { path: '#/chain-of-custody', label: 'Chain of Custody', icon: 'custody' },
    { path: '#/integrity', label: 'Integrity', icon: 'security' },
    { path: '#/face-verification', label: 'Face Verification', icon: 'investigators' },
  ]},
  { group: 'Output', items: [
    { path: '#/reports', label: 'Reports', icon: 'reports' },
  ]},
  { group: 'Account', items: [
    { path: '#/profile', label: 'Profile', icon: 'profile' },
  ]},
];
const NAV_ADMIN = [
  { group: 'Overview', items: [
    { path: '#/admin', label: 'Dashboard', icon: 'dashboard' },
  ]},
  { group: 'Directory', items: [
    { path: '#/admin-users', label: 'Users', icon: 'users' },
    { path: '#/admin-investigators', label: 'Investigators', icon: 'investigators' },
  ]},
  { group: 'Forensic Data', items: [
    { path: '#/admin-cases', label: 'Cases', icon: 'cases' },
    { path: '#/admin-evidence', label: 'Evidence', icon: 'evidence' },
    { path: '#/admin-activity', label: 'Analysis Activity', icon: 'activity' },
  ]},
  { group: 'Oversight', items: [
    { path: '#/admin-reports', label: 'Reports', icon: 'reports' },
    { path: '#/admin-logs', label: 'System Logs', icon: 'logs' },
    { path: '#/admin-security', label: 'Security', icon: 'security' },
    { path: '#/admin-settings', label: 'Settings', icon: 'settings' },
  ]},
];

/* Forensic tools share one evidence id. Sidebar links otherwise drop ?id=
   and every item opens the same case picker. */
function withCurrentEvidence(path) {
  const params = window.__routeParams;
  if (!params || typeof params.get !== 'function') return path;
  const id = params.get('id');
  if (!id) return path;
  const carry = {
    '#/analysis': true,
    '#/xai': true,
    '#/tampering': true,
    '#/chain-of-custody': true,
    '#/integrity': true,
    '#/face-verification': true,
  };
  if (!carry[path]) return path;
  return path + '?id=' + encodeURIComponent(id);
}

function sidebarHtml(role, activePath) {
  const nav = role === 'admin' ? NAV_ADMIN : NAV_INVESTIGATOR;
  const auth = getAuth();
  const groups = nav.map(g => `
    <div class="sidebar-section-label">${g.group}</div>
    ${g.items.map(it => `
      <a href="${role === 'admin' ? it.path : withCurrentEvidence(it.path)}" class="side-link ${activePath===it.path?'active':''}">
        ${ic(it.icon)} <span>${it.label}</span>
      </a>
    `).join('')}
  `).join('');

  return `
  <aside class="sidebar" id="sidebar">
    <div class="sidebar-head">
      <div class="brand"><span class="brand-mark">${ICONS.logo}</span><div>EVIDEX<small>${role==='admin'?'ADMIN CONSOLE':'FORENSIC WORKSPACE'}</small></div></div>
    </div>
    <nav class="sidebar-nav">${groups}</nav>
    <div class="sidebar-foot">
      <div class="user-chip">
        <div class="user-avatar">${initials(auth?.name||'U')}</div>
        <div><div class="uname">${escapeHtml(auth?.name||'')}</div><div class="urole">${role==='admin'?'Administrator':'Investigator'}</div></div>
      </div>
      <a href="#" class="side-link" data-action="logout" style="margin-top:6px;">${ic('logout')} <span>Logout</span></a>
    </div>
  </aside>`;
}

function topbarHtml(title) {
  return `
  <header class="topbar">
    <div style="display:flex;align-items:center;gap:14px;">
      <button class="btn-icon hamburger" data-action="toggle-sidebar" aria-label="Menu">${ic('menu')}</button>
      <h1>${escapeHtml(title)}</h1>
    </div>
    <div class="topbar-actions">
      <span class="pill pill-green">${ic('checkCircle')} System Online</span>
    </div>
  </header>`;
}

function appShell(role, activePath, title, innerHtml) {
  return `
  <div class="app-shell">
    ${sidebarHtml(role, activePath)}
    <div class="main-col">
      ${topbarHtml(title)}
      <div class="page">${innerHtml}</div>
    </div>
  </div>`;
}

/* ==========================================================================
   7. ROUTER
   ========================================================================== */
const ROUTES = {
  '#/home': { render: () => renderLanding(), guard: null },
  '#/login': { render: () => renderLogin(), guard: null },
  '#/public': { render: () => renderPublicUpload(), guard: null },
  '#/public-result': { render: () => renderPublicResult(), guard: null },

  '#/dashboard': { render: () => renderDashboard(), guard: 'investigator' },
  '#/upload': { render: () => renderUpload(), guard: 'investigator' },
  '#/analysis': { render: () => renderAnalysisRoute(), guard: 'investigator' },
  '#/xai': { render: () => renderXAI(), guard: 'investigator' },
  '#/tampering': { render: () => renderTampering(), guard: 'investigator' },
  '#/chain-of-custody': { render: () => renderCustodyLock(), guard: 'investigator' },
  '#/custody-report': { render: () => renderCustodyReport(), guard: 'investigator' },
  '#/cases': { render: () => renderCases(), guard: 'investigator' },
  '#/case-detail': { render: () => renderCaseDetail(), guard: 'investigator' },
  '#/reports': { render: () => renderReports('investigator'), guard: 'investigator' },
  '#/integrity': { render: () => renderIntegrityRoute(), guard: 'investigator' },
  '#/face-verification': { render: () => renderFaceVerificationRoute(), guard: 'investigator' },
  '#/profile': { render: () => renderProfile(), guard: 'investigator' },

  '#/admin': { render: () => renderAdminDashboard(), guard: 'admin' },
  '#/admin-users': { render: () => renderAdminUsers('All') , guard: 'admin' },
  '#/admin-investigators': { render: () => renderAdminUsers('Investigator'), guard: 'admin' },
  '#/admin-cases': { render: () => renderAdminCases(), guard: 'admin' },
  '#/admin-evidence': { render: () => renderAdminEvidence(), guard: 'admin' },
  '#/admin-activity': { render: () => renderAdminActivity(), guard: 'admin' },
  '#/admin-reports': { render: () => renderReports('admin'), guard: 'admin' },
  '#/admin-logs': { render: () => renderAdminLogs(), guard: 'admin' },
  '#/admin-security': { render: () => renderAdminSecurity(), guard: 'admin' },
  '#/admin-settings': { render: () => renderAdminSettings(), guard: 'admin' },
};

function parseHash() {
  const raw = location.hash || '#/home';
  const [path, qs] = raw.split('?');
  const params = new URLSearchParams(qs || '');
  return { path, params };
}

async function router() {
  const { path, params } = parseHash();
  window.__routeParams = params;

  // Reset the global modal before every route render. The modal lives
  // outside #app, so changing routes does not automatically remove it.
  // Without this reset, an open/empty modal can remain over the page and
  // keep the entire background blurred.
  const modalOverlay = $('#modalOverlay');
  const modalBox = $('#modalBox');
  if (modalOverlay) modalOverlay.classList.remove('show');
  if (modalBox) modalBox.innerHTML = '';

  const entry = ROUTES[path];
  const app = $('#app');

  if (!entry) { location.hash = '#/home'; return; }

  if (entry.guard) {
    const auth = requireAuth(entry.guard);
    if (!auth) return;
  }

  document.body.classList.remove('sidebar-open');
  window.scrollTo(0, 0);

  // Renderers that talk to the backend are async; the demo pages stay sync and
  // `await` passes their plain strings straight through.
  let html;
  try {
    html = await entry.render();
  } catch (err) {
    handleApiError(err, 'Could not load page');
    html = `<div class="page"><div class="card"><div class="panel-title"><h3>Page failed to load</h3></div>
      <p style="color:var(--text-muted);font-size:13.5px;">${escapeHtml(err.message || String(err))}</p></div></div>`;
  }

  // Ignore a stale response if the user navigated again while we were loading.
  if (parseHash().path !== path) return;

  app.innerHTML = `<div class="view active">${html}</div>`;
  if (typeof window.__postRender === 'function') { window.__postRender(); window.__postRender = null; }
}

window.addEventListener('hashchange', router);

/* ==========================================================================
   8. LANDING PAGE
   ========================================================================== */
function renderLanding() {
  window.__postRender = postRenderLanding;
  const nodes = [
    { top: '8%', left: '50%' }, { top: '30%', left: '85%' }, { top: '68%', left: '82%' },
    { top: '88%', left: '48%' }, { top: '68%', left: '14%' }, { top: '30%', left: '12%' },
  ];
  return `
  <nav class="nav">
    <a href="#/home" class="brand"><span class="brand-mark">${ICONS.logo}</span><div>EVIDEX<small>DIGITAL EVIDENCE VERIFICATION</small></div></a>
    <div class="nav-links">
      <a href="#features-anchor" data-scroll>Platform</a>
      <a href="#roles-anchor" data-scroll>Who it's for</a>
      <a href="#features-anchor" data-scroll>Capabilities</a>
    </div>
    <div class="nav-actions">
      <a href="#/login" class="btn btn-ghost btn-sm">Login</a>
      <a href="#/login" class="btn btn-primary btn-sm">Investigator Login</a>
    </div>
  </nav>

  <section class="hero">
    <div>
      <div class="hero-eyebrow"><span class="dot"></span> AI-assisted forensic verification</div>
      <h1>Know whether digital evidence is <span class="grad">authentic</span> before it reaches a courtroom.</h1>
      <p class="lead">EVIDEX analyzes images, video, audio and documents for tampering and AI generation, explains its reasoning in plain terms, and keeps a verifiable chain of custody — from the first upload to the final report.</p>
      <div class="hero-actions">
        <a href="#/login" class="btn btn-primary">Sign In ${ic('arrowRight')}</a>
        <a href="#/login" class="btn btn-ghost">Investigator / Admin</a>
        <a href="#features-anchor" data-scroll class="btn btn-ghost">Explore Platform</a>
      </div>
    </div>
    <div class="hero-visual" aria-hidden="true">
      <div class="scan-ring r1"></div>
      <div class="scan-ring r2"></div>
      <div class="scan-sweep"></div>
      <div class="scan-ring r3"></div>
      <div class="scan-core">Scanning<br>evidence</div>
      ${nodes.map(n => `<div class="scan-node" style="top:${n.top};left:${n.left};"></div>`).join('')}
    </div>
  </section>

  <section class="section alt" id="features-anchor">
    <div class="section-head">
      <div class="section-label">Platform capabilities</div>
      <h2>Every layer a forensic review needs</h2>
      <p>From a first integrity check to a court-ready report, each step is logged, explainable and repeatable.</p>
    </div>
    <div class="feature-grid">
      ${[
        ['evidence','Evidence Verification','Upload images, video, audio or documents and get a structured authenticity assessment.'],
        ['hash','SHA-256 Integrity','Every file is fingerprinted on intake so any later modification is detectable.'],
        ['layers','Metadata Analysis','Inspects embedded metadata for inconsistencies that point to editing or re-export.'],
        ['tampering','Tampering Detection','Flags manipulated regions with confidence scores and severity ratings.'],
        ['scan','Deepfake Detection','Screens faces and voices for synthetic-generation artifacts.'],
        ['xai','Explainable AI','Shows the reasoning behind a verdict, not just a black-box score.'],
        ['custody','Chain of Custody','Tracks every handoff of the evidence from acquisition to reporting.'],
        ['gavel','Legal Reports','Generates admissibility-oriented documentation for investigators and counsel.'],
      ].map(([icon,title,desc]) => `
        <div class="feature-card">
          <div class="fi">${ic(icon)}</div>
          <h3>${title}</h3>
          <p>${desc}</p>
        </div>`).join('')}
    </div>
  </section>

  <section class="section">
    <div class="stats-strip">
      <div><div class="num" data-count="18420">0</div><div class="lbl">Files verified</div></div>
      <div><div class="num" data-count="97" data-suffix="%">0</div><div class="lbl">Detection confidence avg.</div></div>
      <div><div class="num" data-count="640">0</div><div class="lbl">Active cases supported</div></div>
      <div><div class="num" data-count="24" data-suffix="/7">0</div><div class="lbl">Continuous monitoring</div></div>
    </div>
  </section>

  <section class="section alt" id="roles-anchor">
    <div class="section-head">
      <div class="section-label">Built for investigators and admins</div>
      <h2>One platform, the right depth for each role</h2>
    </div>
    <div class="role-grid">
      <div class="role-card">
        <div class="rc-icon" style="background:var(--cyan-dim);color:var(--cyan);">${ic('analysis')}</div>
        <h3>Investigator</h3>
        <p>Full forensic workspace: case management, XAI reasoning, tampering heat maps and chain-of-custody records.</p>
        <a href="#/login" class="btn btn-ghost btn-sm">Investigator login</a>
      </div>
      <div class="role-card">
        <div class="rc-icon" style="background:var(--amber-dim);color:var(--amber);">${ic('security')}</div>
        <h3>Admin</h3>
        <p>System oversight: manage users and investigators, monitor activity, and review security events.</p>
        <a href="#/login" class="btn btn-ghost btn-sm">Admin login</a>
      </div>
    </div>
  </section>

  <footer class="site-footer">
    <div class="footer-grid">
      <div>
        <div class="brand"><span class="brand-mark">${ICONS.logo}</span><div>EVIDEX</div></div>
        <p style="color:var(--text-dim);font-size:13px;margin-top:14px;max-width:280px;">Intelligent digital evidence verification for forensic investigation and case management.</p>
      </div>
      <div><h4>Platform</h4><ul>
        <li><a href="#features-anchor" data-scroll>Capabilities</a></li>
        <li><a href="#/login">Login</a></li>
      </ul></div>
      <div><h4>Roles</h4><ul>
        <li><a href="#/login">Investigator</a></li>
        <li><a href="#/login">Admin</a></li>
      </ul></div>
      <div><h4>Project</h4><ul>
        <li>BTech Final Year Project</li>
        <li>AI-Based Digital Evidence Authenticity Verification</li>
      </ul></div>
    </div>
    <div class="footer-bottom">
      <span>© 2026 EVIDEX. Academic demonstration project — not a certified forensic tool.</span>
      <span>Built as a frontend-only prototype</span>
    </div>
  </footer>
  `;
}

/* Animate landing stat counters + smooth-scroll anchors (runs after landing mounts) */
function postRenderLanding() {
  $$('[data-scroll]').forEach(a => {
    a.addEventListener('click', (e) => {
      const targetId = a.getAttribute('data-scroll') || a.getAttribute('href').replace('#','');
      const target = document.getElementById(targetId);
      if (target) { e.preventDefault(); target.scrollIntoView({ behavior: 'smooth' }); }
    });
  });
  $$('.num[data-count]').forEach(el => {
    const target = parseInt(el.dataset.count, 10);
    const suffix = el.dataset.suffix || '';
    let cur = 0;
    const step = Math.max(1, Math.round(target / 40));
    const t = setInterval(() => {
      cur += step;
      if (cur >= target) { cur = target; clearInterval(t); }
      el.textContent = cur.toLocaleString() + suffix;
    }, 25);
  });
}

/* ==========================================================================
   9. LOGIN
   ========================================================================== */
function renderLogin() {
  window.__postRender = postRenderLogin;
  return `
  <div class="auth-shell">
    <div class="auth-card card glass">
      <div class="brand"><span class="brand-mark">${ICONS.logo}</span><div>EVIDEX</div></div>
      <p class="auth-sub">Sign in to continue to your workspace</p>

      <div class="role-select" id="roleSelect">
        <button type="button" class="role-opt active" data-role="investigator">${ic('investigators')}<span>Investigator</span></button>
        <button type="button" class="role-opt" data-role="admin">${ic('security')}<span>Admin</span></button>
      </div>

      <div class="form-error" id="loginError">Invalid username or password for the selected role.</div>

      <form id="loginForm">
        <div class="field">
          <label for="loginUser">Username</label>
          <input type="text" id="loginUser" autocomplete="username" placeholder="Enter username">
        </div>
        <div class="field" id="passwordField">
          <label for="loginPass">Password</label>
          <div class="password-row">
            <input type="password" id="loginPass" autocomplete="current-password" placeholder="Enter password">
            <button type="button" class="btn btn-ghost btn-sm password-toggle" data-toggle-password="loginPass">Show</button>
          </div>
        </div>
        <button type="submit" class="btn btn-primary btn-block" id="loginSubmitBtn">Sign In as Investigator</button>
      </form>

      <div class="demo-box" id="demoBox">
        <strong>MAYA investigator account required.</strong><br>Register below, or sign in with an existing account.
      </div>

      <div class="divider"></div>
      <p class="auth-sub" style="margin:0 0 10px;">No account yet?</p>
      <button type="button" class="btn btn-ghost btn-block" id="showRegisterBtn">Create an investigator account</button>

      <form id="registerForm" style="display:none;margin-top:14px;">
        <div class="field">
          <label for="regFullName">Full name</label>
          <input type="text" id="regFullName" autocomplete="name" placeholder="Jane Investigator">
        </div>
        <div class="field">
          <label for="regUsername">Username</label>
          <input type="text" id="regUsername" autocomplete="username" placeholder="At least 3 characters">
        </div>
        <div class="field">
          <label for="regEmail">Email</label>
          <input type="email" id="regEmail" autocomplete="email" placeholder="you@agency.gov">
        </div>
        <div class="field">
          <label for="regPassword">Password</label>
          <div class="password-row">
            <input type="password" id="regPassword" autocomplete="new-password" placeholder="Enter a password">
            <button type="button" class="btn btn-ghost btn-sm password-toggle" data-toggle-password="regPassword">Show</button>
          </div>
          <div class="pw-reqs-title">Password requirements</div>
          <ul class="pw-reqs" id="regPasswordReqs">
            <li data-rule="length" data-label="At least 8 characters">☐ At least 8 characters</li>
            <li data-rule="uppercase" data-label="1 uppercase letter">☐ 1 uppercase letter</li>
            <li data-rule="lowercase" data-label="1 lowercase letter">☐ 1 lowercase letter</li>
            <li data-rule="number" data-label="1 number">☐ 1 number</li>
            <li data-rule="special" data-label="1 special character">☐ 1 special character</li>
          </ul>
        </div>
        <button type="submit" class="btn btn-primary btn-block" id="registerSubmitBtn" disabled>Register &amp; Sign In</button>
      </form>
    </div>
  </div>`;
}

function postRenderLogin() {
  let role = 'investigator';
  const roleSelect = $('#roleSelect');
  const demoBox = $('#demoBox');
  const submitBtn = $('#loginSubmitBtn');
  const userInput = $('#loginUser');

  function applyRole(r) {
    role = r;
    $$('.role-opt', roleSelect).forEach(b => b.classList.toggle('active', b.dataset.role === r));
    $('#loginError').classList.remove('show');
    if (r === 'investigator') {
      submitBtn.textContent = 'Sign In as Investigator';
      demoBox.innerHTML = `<strong>MAYA account required.</strong><br>Sign in with your registered MAYA username or email.`;
      userInput.placeholder = 'username or email';
    } else {
      submitBtn.textContent = 'Sign In as Admin';
      demoBox.innerHTML = `<strong>MAYA ADMIN account required.</strong><br>Use an account whose backend role is ADMIN.`;
      userInput.placeholder = 'username or email';
    }
  }

  $$('.role-opt', roleSelect).forEach(btn => btn.addEventListener('click', () => applyRole(btn.dataset.role)));
  applyRole('investigator');

  const regPassword = $('#regPassword');
  const regButton = $('#registerSubmitBtn');
  function refreshPasswordRequirements() {
    if (!window.MayaPassword || !regPassword || !regButton) return;
    const username = ($('#regUsername') && $('#regUsername').value) || '';
    const email = ($('#regEmail') && $('#regEmail').value) || '';
    MayaPassword.applyChecklist(regPassword.value, $$('#regPasswordReqs [data-rule]'), null);
    regButton.disabled = !MayaPassword.registrationReady({
      password: regPassword.value,
      username: username,
      email: email,
    });
  }
  $$('[data-toggle-password]').forEach(button => {
    button.addEventListener('click', () => {
      const input = document.getElementById(button.getAttribute('data-toggle-password'));
      if (!input) return;
      const visible = input.type === 'text';
      input.type = visible ? 'password' : 'text';
      button.textContent = visible ? 'Show' : 'Hide';
      if (input === regPassword) refreshPasswordRequirements();
    });
  });
  ['regPassword', 'regUsername', 'regEmail'].forEach(id => {
    const field = document.getElementById(id);
    if (!field) return;
    field.addEventListener('input', refreshPasswordRequirements);
    field.addEventListener('change', refreshPasswordRequirements);
  });
  refreshPasswordRequirements();

  const errBox = $('#loginError');
  function showLoginError(message) {
    errBox.textContent = message;
    errBox.classList.add('show');
  }

  /* Real login: Flask-Login issues a session cookie. The selected role is
     verified against the role the backend actually returns — the client can
     never grant itself admin. */
  $('#loginForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const username = $('#loginUser').value.trim();
    const password = $('#loginPass').value;

    errBox.classList.remove('show');
    submitBtn.disabled = true;
    submitBtn.textContent = 'Signing in…';
    try {
      const user = await MayaApi.auth.login(username, password);
      const auth = authFromBackendUser(user);
      if (role === 'admin' && auth.role !== 'admin') {
        await MayaApi.auth.logout().catch(() => {});
        clearAuth();
        showLoginError('This account does not have the ADMIN role.');
        return;
      }
      setAuth(auth);
      toast('Signed in', `Welcome back, ${auth.name}.`, 'success');
      location.hash = auth.role === 'admin' ? '#/admin' : '#/dashboard';
    } catch (err) {
      if (err instanceof MayaApi.ApiError) showLoginError(err.message);
      else showLoginError('Cannot reach the MAYA backend.');
    } finally {
      submitBtn.disabled = false;
      applyRole(role);
    }
  });

  /* Registration uses the existing /api/auth/register endpoint, then logs in. */
  const registerForm = $('#registerForm');
  $('#showRegisterBtn').addEventListener('click', () => {
    const shown = registerForm.style.display === 'block';
    registerForm.style.display = shown ? 'none' : 'block';
  });

  registerForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = $('#registerSubmitBtn');
    const payload = {
      full_name: $('#regFullName').value.trim(),
      username: $('#regUsername').value.trim(),
      email: $('#regEmail').value.trim(),
      password: $('#regPassword').value,
    };
    if (!window.MayaPassword || !MayaPassword.satisfied(MayaPassword.checks(payload.password))) {
      showLoginError('Password must be at least 8 characters and include an uppercase letter, a lowercase letter, a number, and a special character.');
      return;
    }
    errBox.classList.remove('show');
    btn.disabled = true;
    btn.textContent = 'Creating account…';
    try {
      await MayaApi.auth.register(payload);
      const user = await MayaApi.auth.login(payload.username, payload.password);
      setAuth(authFromBackendUser(user));
      toast('Account created', 'You are now signed in.', 'success');
      location.hash = '#/dashboard';
    } catch (err) {
      if (err instanceof MayaApi.ApiError) showLoginError(err.message);
      else showLoginError('Cannot reach the MAYA backend.');
    } finally {
      btn.textContent = 'Register & Sign In';
      refreshPasswordRequirements();
    }
  });
}

/* ==========================================================================
   10. PUBLIC FLOW  (simple upload -> basic result, no forensic detail)
   ========================================================================== */
function stepsHtml(activeIndex) {
  const steps = ['Upload Evidence', 'Analysis', 'Basic Result'];
  return `<div class="public-steps">${steps.map((s,i) => `
    <div class="public-step ${i===activeIndex?'active':i<activeIndex?'done':''}">
      <span class="num">${i<activeIndex? ic('check') : i+1}</span><span>${s}</span>
    </div>
    ${i<steps.length-1?'<span class="step-sep"></span>':''}`).join('')}</div>`;
}

function renderPublicUpload() {
  const auth = getAuth();
  window.__postRender = postRenderPublicUpload;
  return `
  <nav class="nav">
    <a href="#/home" class="brand"><span class="brand-mark">${ICONS.logo}</span><div>EVIDEX</div></a>
    <div class="nav-actions">
      ${auth ? `<a href="#" class="btn btn-ghost btn-sm" data-action="logout">Exit</a>` : `<a href="#/login" class="btn btn-ghost btn-sm">Login</a>`}
    </div>
  </nav>
  <div class="public-shell">
    ${demoBanner('Public verification')}
    ${stepsHtml(0)}
    <div class="card glass" style="padding:8px;">
      <div style="padding:20px 20px 4px;">
        <h2 style="font-size:20px;margin-bottom:6px;">Check a file's authenticity</h2>
        <p style="color:var(--text-muted);font-size:13.5px;">
          This public preview runs entirely in your browser and does <strong>not</strong> use the MAYA model.
          Real analysis requires an investigator account — <a href="#/login" style="color:var(--cyan);">sign in</a> to analyse evidence.
        </p>
      </div>
      <div style="padding:20px;">
        <div class="dropzone" id="publicDropzone" tabindex="0">
          <div class="dz-icon">${ic('upload')}</div>
          <h3>Drag & drop a file here</h3>
          <p>or click to browse from your device</p>
          <div class="type-chips">
            <span class="type-chip">${ic('image')} Images</span>
            <span class="type-chip">${ic('video')} Video</span>
            <span class="type-chip">${ic('audio')} Audio</span>
            <span class="type-chip">${ic('doc')} Documents</span>
          </div>
          <input type="file" id="publicFileInput">
        </div>
      </div>
    </div>
  </div>`;
}

function fileKind(file) {
  if (file.type.startsWith('image/')) return 'Image';
  if (file.type.startsWith('video/')) return 'Video';
  if (file.type.startsWith('audio/')) return 'Audio';
  return 'Document';
}
function fileIconFor(kind) { return { Image:'image', Video:'video', Audio:'audio', Document:'doc' }[kind] || 'doc'; }

async function processSelectedFile(file, targetHashHtmlId, onDone) {
  const kind = fileKind(file);
  const buf = await file.arrayBuffer();
  const hash = await sha256Hex(new Uint8Array(buf));
  const evId = uid('EVX');
  const record = {
    id: evId, filename: file.name, type: kind, size: file.size,
    sha256: hash, timestamp: new Date().toISOString(),
    mimeType: file.type,
  };
  setCtx({ pendingEvidence: record });
  if (file.type.startsWith('image/')) {
    const reader = new FileReader();
    reader.onload = () => { setCtx({ pendingEvidence: { ...record, previewDataUrl: reader.result } }); onDone(record); };
    reader.readAsDataURL(file);
  } else {
    onDone(record);
  }
}

function postRenderPublicUpload() {
  const dz = $('#publicDropzone');
  const input = $('#publicFileInput');
  dz.addEventListener('click', () => input.click());
  dz.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') input.click(); });
  ['dragenter','dragover'].forEach(evt => dz.addEventListener(evt, (e) => { e.preventDefault(); dz.classList.add('drag'); }));
  ['dragleave','drop'].forEach(evt => dz.addEventListener(evt, (e) => { e.preventDefault(); dz.classList.remove('drag'); }));
  dz.addEventListener('drop', (e) => { const f = e.dataTransfer.files[0]; if (f) handleFile(f); });
  input.addEventListener('change', (e) => { const f = e.target.files[0]; if (f) handleFile(f); });

  function handleFile(f) {
    toast('File received', `Analyzing ${f.name}…`, 'success', 2200);
    processSelectedFile(f, null, () => { location.hash = '#/public-result'; });
  }
}

function renderPublicResult() {
  const ctx = getCtx();
  const ev = ctx.pendingEvidence;
  if (!ev) {
    return `<div class="public-shell"><div class="empty-state card">No file uploaded yet. <a href="#/public" style="color:var(--cyan);">Upload one to check its authenticity</a>.</div></div>`;
  }
  window.__postRender = () => postRenderPublicResult(ev);
  return `
  <nav class="nav">
    <a href="#/home" class="brand"><span class="brand-mark">${ICONS.logo}</span><div>EVIDEX</div></a>
    <div class="nav-actions"><a href="#/public" class="btn btn-ghost btn-sm">Check another file</a></div>
  </nav>
  <div class="public-shell">
    ${stepsHtml(2)}
    <div class="card glass" id="publicResultCard" style="padding:26px;">
      <div style="display:flex;gap:16px;align-items:center;">
        <div class="file-thumb" style="width:52px;height:52px;">${ev.previewDataUrl ? `<img src="${ev.previewDataUrl}" alt="">` : ic(fileIconFor(ev.type))}</div>
        <div style="min-width:0;">
          <div style="font-weight:600;font-size:14.5px;word-break:break-all;">${escapeHtml(ev.filename)}</div>
          <div style="font-size:12px;color:var(--text-dim);">${ev.type} · ${fmtBytes(ev.size)} · ${fmtDate(ev.timestamp)}</div>
        </div>
      </div>

      <div id="publicProgressArea" style="margin-top:26px;">
        <div class="progress-list" id="publicProgressList"></div>
      </div>

      <div id="publicVerdictArea" style="display:none;"></div>
    </div>
  </div>`;
}

function postRenderPublicResult(ev) {
  const steps = ['Checking file integrity', 'Reading metadata', 'Scanning for manipulation', 'Finalizing result'];
  const listEl = $('#publicProgressList');
  listEl.innerHTML = steps.map((s,i) => `
    <div class="progress-step" data-i="${i}">
      <div class="ps-icon">${i+1}</div>
      <div class="ps-label">${s}</div>
      <div class="ps-bar"><span></span></div>
    </div>`).join('');

  let i = 0;
  function runStep() {
    if (i >= steps.length) { finishPublic(ev); return; }
    const row = listEl.querySelector(`[data-i="${i}"]`);
    row.classList.add('active');
    const bar = row.querySelector('.ps-bar span');
    requestAnimationFrame(() => bar.style.width = '100%');
    setTimeout(() => {
      row.classList.remove('active'); row.classList.add('done');
      row.querySelector('.ps-icon').innerHTML = ic('check');
      i++; runStep();
    }, 550);
  }
  runStep();

  function finishPublic(ev) {
    const rand = seededRandom(ev.sha256 || ev.id);
    const score = Math.round((38 + rand() * 60) * 10) / 10;
    let verdict = 'LIKELY AUTHENTIC', tone = 'green', explain = 'No significant signs of manipulation were found. File integrity and metadata are consistent.';
    if (score < 55) { verdict = 'POTENTIALLY MANIPULATED'; tone = 'red'; explain = 'Several inconsistencies were found in this file that are commonly associated with editing or manipulation.'; }
    else if (score < 74) { verdict = 'INCONCLUSIVE'; tone = 'amber'; explain = 'The check could not confidently confirm authenticity. Some signals were ambiguous or the file quality limited analysis.'; }

    setCtx({ pendingEvidence: { ...ev, score, verdict, tone } });

    $('#publicVerdictArea').style.display = 'block';
    $('#publicVerdictArea').innerHTML = `
      <div class="divider"></div>
      <div class="result-hero">
        <div class="score-ring">
          <svg viewBox="0 0 120 120">
            <circle class="bg" cx="60" cy="60" r="52"></circle>
            <circle class="fg" cx="60" cy="60" r="52" style="stroke:var(--${tone});stroke-dasharray:${2*Math.PI*52};stroke-dashoffset:${2*Math.PI*52};"></circle>
          </svg>
          <div class="val"><div class="n">${score}%</div><div class="l">AUTHENTICITY</div></div>
        </div>
        <div class="rh-badge" style="color:var(--${tone});">${verdict}</div>
      </div>
      <p style="text-align:center;color:var(--text-muted);font-size:13.5px;max-width:440px;margin:8px auto 0;">${explain}</p>

      <div class="kv-grid" style="margin-top:26px;">
        <div class="kv-item"><div class="kl">Authenticity score</div><div class="kv-val">${score}%</div></div>
        <div class="kv-item"><div class="kl">File integrity</div><div class="kv-val">${ic('checkCircle')} Hash recorded</div></div>
        <div class="kv-item"><div class="kl">Checked on</div><div class="kv-val">${fmtDate(new Date())}</div></div>
        <div class="kv-item"><div class="kl">SHA-256 (first 16)</div><div class="kv-val">${(ev.sha256||'').slice(0,16)}…</div></div>
      </div>

      <div class="disclaimer-box">${ic('helpCircle')} For legal or forensic investigation, sign in as an <a href="#/login" style="color:var(--cyan);">Investigator</a>. This basic check does not include chain of custody, advanced explainability or admissibility review.</div>

      <div style="display:flex;gap:10px;margin-top:20px;flex-wrap:wrap;">
        <button class="btn btn-primary btn-sm" data-action="download-public-result">${ic('download')} Download this result</button>
        <a href="#/public" class="btn btn-ghost btn-sm">Check another file</a>
      </div>
    `;
    requestAnimationFrame(() => {
      const fg = $('#publicVerdictArea .fg');
      if (fg) fg.style.strokeDashoffset = (2*Math.PI*52) * (1 - score/100);
    });
  }
}

/* ==========================================================================
   11. INVESTIGATOR DASHBOARD
   ========================================================================== */
/* Also accepts the backend's own status enums, since this now renders values
   that came from the API rather than a fixed local vocabulary. */
function statusPill(status) {
  const map = {
    Authentic: 'green', Tampered: 'red', Unavailable: 'muted',
    // Evidence / analysis status values from MAYA
    COMPLETED: 'green', UPLOADED: 'blue', VERIFIED: 'green',
    PROCESSING: 'amber', PENDING: 'amber', NONE: 'muted',
    FAILED: 'red', REAL: 'green', FAKE: 'red',
    'Not analysed': 'muted', 'Analysis failed': 'red',
  };
  return `<span class="pill pill-${map[status]||'muted'}">${escapeHtml(status)}</span>`;
}
function riskPill(risk) {
  const map = { High:'red', Medium:'amber', Low:'green' };
  return `<span class="pill pill-${map[risk]||'muted'}">${escapeHtml(risk)}</span>`;
}

/* Dashboard is driven entirely by GET /api/dashboard/stats — real ownership
   scoped counts and real timestamp-derived trends. */
async function renderDashboard() {
  const auth = getAuth();
  const stats = await MayaApi.dashboard.stats();
  window.__dashboardStats = stats;
  window.__postRender = postRenderDashboard;

  const counts = stats.counts;
  const pred = stats.prediction || {};
  const authentic = pred.REAL || 0;
  const tampered = pred.FAKE || 0;
  const analysed = counts.analyses_completed || 0;
  const notAnalysed = Math.max(0, counts.evidence - analysed);
  const pct = (n) => (counts.evidence ? Math.round((n / counts.evidence) * 100) : 0);
  const openCases = stats.case_status?.OPEN || 0;

  const recent = stats.recent_analyses || [];

  const inner = `
    <p class="page-sub">Welcome back, ${escapeHtml(auth.name)} — live figures from your MAYA workspace${stats.scope === 'all' ? ' (all users)' : ''}.</p>
    <div class="stat-grid">
      <div class="stat-card"><div class="stat-label">Total Cases</div><div class="stat-value">${counts.cases}</div><div class="stat-delta">${ic('arrowRight')} ${openCases} open</div></div>
      <div class="stat-card tone-green"><div class="stat-label">Authentic (REAL)</div><div class="stat-value">${authentic}</div><div class="stat-delta">${pct(authentic)}% of evidence</div></div>
      <div class="stat-card tone-red"><div class="stat-label">Tampered (FAKE)</div><div class="stat-value">${tampered}</div><div class="stat-delta down">${pct(tampered)}% of evidence</div></div>
      <div class="stat-card tone-amber"><div class="stat-label">Not Yet Analysed</div><div class="stat-value">${notAnalysed}</div><div class="stat-delta">of ${counts.evidence} evidence items</div></div>
      <div class="stat-card tone-blue"><div class="stat-label">Reports Generated</div><div class="stat-value">${counts.reports}</div><div class="stat-delta">${counts.face_verifications} face checks</div></div>
    </div>

    <div class="grid-charts">
      <div class="card">
        <div class="panel-title"><h3>Authenticity Trend</h3><span class="sub">Completed analyses, last 14 days</span></div>
        <div class="chart-wrap"><canvas id="chartTrend"></canvas></div>
      </div>
      <div class="card">
        <div class="panel-title"><h3>Authentic vs Tampered</h3><span class="sub">Completed analyses</span></div>
        <div class="chart-wrap"><canvas id="chartAuthVsTamper"></canvas></div>
      </div>
      <div class="card">
        <div class="panel-title"><h3>Case Activity</h3><span class="sub">Cases opened per week</span></div>
        <div class="chart-wrap"><canvas id="chartCaseActivity"></canvas></div>
      </div>
      <div class="card">
        <div class="panel-title"><h3>Evidence Status</h3><span class="sub">Across your evidence</span></div>
        <div class="chart-wrap"><canvas id="chartRisk"></canvas></div>
      </div>
    </div>

    <div class="card">
      <div class="panel-title"><h3>Recent Analyses</h3><a href="#/cases" class="btn btn-ghost btn-sm">View all cases</a></div>
      <div class="table-wrap">
        <table class="data-table">
          <thead><tr><th>Investigation</th><th>Evidence</th><th>Case</th><th>Status</th><th>Authenticity</th><th>Confidence</th><th>Completed</th><th></th></tr></thead>
          <tbody>
            ${recent.length ? recent.map(r => {
              const verdict = MayaApi.adapt.verdictFromPrediction(r.prediction, r.confidence);
              return `
              <tr>
                <td class="id-cell">${escapeHtml(r.investigation_id || '—')}</td>
                <td class="id-cell">EV-${r.evidence_id}</td>
                <td class="id-cell">CASE-${r.case_id}</td>
                <td>${statusPill(r.status === 'COMPLETED' ? verdict : r.status)}</td>
                <td>${r.prediction ? escapeHtml(r.prediction) : '—'}</td>
                <td>${typeof r.confidence === 'number' ? r.confidence.toFixed(1) + '%' : '—'}</td>
                <td>${r.completed_at ? fmtDateShort(r.completed_at) : '—'}</td>
                <td class="row-actions"><a href="#/analysis?id=${r.evidence_id}" class="btn btn-ghost btn-sm">Open</a></td>
              </tr>`;
            }).join('') : `<tr><td colspan="8" style="color:var(--text-muted);">No analyses yet — upload evidence to begin.</td></tr>`}
          </tbody>
        </table>
      </div>
    </div>
  `;
  return appShell('investigator', '#/dashboard', 'Dashboard', inner);
}

function postRenderDashboard() {
  const stats = window.__dashboardStats;
  if (!stats || typeof Chart === 'undefined') return;
  const grid = { color: '#1e2a38' };
  const ticks = { color: '#5b6c7d', font: { family: 'Inter', size: 11 } };

  const trend = stats.trend || [];
  const labels = trend.map(t => new Date(t.date + 'T00:00:00')
    .toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
  new Chart($('#chartTrend'), {
    type: 'line',
    data: { labels, datasets: [
      { label: 'Authentic', data: trend.map(t => t.real), borderColor:'#35d399', backgroundColor:'rgba(53,211,153,0.12)', fill:true, tension:0.35, pointRadius:0 },
      { label: 'Tampered', data: trend.map(t => t.fake), borderColor:'#f1495b', backgroundColor:'rgba(241,73,91,0.1)', fill:true, tension:0.35, pointRadius:0 },
    ]},
    options: { responsive:true, maintainAspectRatio:false, plugins:{legend:{labels:{color:'#8ca0b3',boxWidth:10,font:{size:11}}}}, scales:{ x:{grid, ticks}, y:{grid, ticks, beginAtZero:true, ticks:{...ticks, precision:0}} } }
  });

  const pred = stats.prediction || {};
  new Chart($('#chartAuthVsTamper'), {
    type: 'doughnut',
    data: { labels: ['Authentic (REAL)','Tampered (FAKE)'], datasets: [{
      data: [pred.REAL || 0, pred.FAKE || 0],
      backgroundColor: ['#35d399','#f1495b'], borderWidth: 0,
    }]},
    options: { responsive:true, maintainAspectRatio:false, cutout:'68%', plugins:{legend:{position:'bottom',labels:{color:'#8ca0b3',boxWidth:10,font:{size:11}}}} }
  });

  const activity = stats.case_activity || [];
  new Chart($('#chartCaseActivity'), {
    type: 'bar',
    data: {
      labels: activity.map(a => new Date(a.week_start + 'T00:00:00')
        .toLocaleDateString('en-US', { month: 'short', day: 'numeric' })),
      datasets: [{ label:'Cases opened', data: activity.map(a => a.opened), backgroundColor:'#29e0d6', borderRadius:4, maxBarThickness:28 }]
    },
    options: { responsive:true, maintainAspectRatio:false, plugins:{legend:{display:false}}, scales:{x:{grid:{display:false},ticks},y:{grid,ticks:{...ticks, precision:0},beginAtZero:true}} }
  });

  const evStatus = stats.evidence_status || {};
  const evLabels = Object.keys(evStatus);
  new Chart($('#chartRisk'), {
    type: 'polarArea',
    data: { labels: evLabels.length ? evLabels : ['No evidence'], datasets:[{
      data: evLabels.length ? evLabels.map(k => evStatus[k]) : [0],
      backgroundColor: ['rgba(53,211,153,0.55)','rgba(46,143,224,0.55)','rgba(245,166,35,0.55)','rgba(241,73,91,0.55)','rgba(41,224,214,0.55)'], borderWidth:0,
    }]},
    options: { responsive:true, maintainAspectRatio:false, scales:{ r:{ grid:{color:'#1e2a38'}, angleLines:{color:'#1e2a38'}, ticks:{display:false} } }, plugins:{legend:{position:'bottom',labels:{color:'#8ca0b3',boxWidth:10,font:{size:11}}}} }
  });
}

/* ==========================================================================
   12. EVIDENCE UPLOAD (investigator) -> ANALYSIS PROCESSING -> RESULT
   ========================================================================== */
/* MAYA scopes evidence to a case. The file picker advertises the backend's
   image and video extensions; the server remains the authority on validation. */
const ACCEPTED_UPLOAD_EXTS = [
  '.jpg', '.jpeg', '.png', '.bmp', '.webp',
  '.mp4', '.avi', '.mov', '.mkv',
];
const ACCEPTED_UPLOAD_MIMES = [
  'image/jpeg', 'image/png', 'image/bmp', 'image/webp',
  'video/mp4', 'video/x-msvideo', 'video/quicktime', 'video/x-matroska',
];
const ACCEPTED_UPLOAD_ACCEPT = [...ACCEPTED_UPLOAD_EXTS, ...ACCEPTED_UPLOAD_MIMES].join(',');

async function renderUpload() {
  const cases = (await MayaApi.cases.list()).map(MayaApi.adapt.case);
  window.__uploadCases = cases;
  window.__postRender = postRenderUpload;

  const options = cases.length
    ? cases.map(c => `<option value="${c.backendId}">${escapeHtml(c.caseNumber)} — ${escapeHtml(c.name)}</option>`).join('')
    : '';

  const inner = `
    <p class="page-sub">Upload evidence to begin forensic verification. Files are hashed and stored server-side.</p>
    <div class="grid-2">
      <div class="card" id="uploadCardArea">
        <div class="field">
          <label for="uploadCase">Case</label>
          ${cases.length ? `<select id="uploadCase" class="input">${options}</select>`
            : `<p style="color:var(--text-muted);font-size:13.5px;margin:0 0 10px;">You have no cases yet. Create one to attach evidence to.</p>`}
        </div>
        <details style="margin-bottom:14px;">
          <summary style="cursor:pointer;color:var(--text-muted);font-size:13px;">Create a new case instead</summary>
          <div class="field" style="margin-top:10px;">
            <label for="newCaseTitle">Case title</label>
            <input type="text" id="newCaseTitle" placeholder="e.g. Contract Exhibit B — forgery review">
          </div>
          <button class="btn btn-ghost btn-sm" id="createCaseBtn">${ic('cases')} Create case</button>
        </details>

        <div class="dropzone" id="invDropzone" tabindex="0">
          <div class="dz-icon">${ic('upload')}</div>
          <h3>Drag &amp; drop evidence here</h3>
          <p>or click to browse</p>
          <p>Images and videos supported</p>
          <div class="type-chips">
            <span class="type-chip">${ic('image')} JPG</span>
            <span class="type-chip">${ic('image')} PNG</span>
            <span class="type-chip">${ic('image')} BMP</span>
            <span class="type-chip">${ic('image')} WEBP</span>
            <span class="type-chip">${ic('video')} MP4</span>
            <span class="type-chip">${ic('video')} AVI</span>
            <span class="type-chip">${ic('video')} MOV</span>
            <span class="type-chip">${ic('video')} MKV</span>
          </div>
          <input type="file" id="invFileInput" accept="${ACCEPTED_UPLOAD_ACCEPT}">
        </div>
        <p style="color:var(--text-muted);font-size:12.5px;margin:10px 0 0;">
          Supported evidence: images (JPG, PNG, BMP, WEBP) and videos (MP4, AVI, MOV, MKV). The server validates each file.
        </p>
        <div id="invFilePreview"></div>
      </div>
      <div class="card">
        <div class="panel-title"><h3>What happens next</h3></div>
        <div class="progress-list" style="margin:4px 0 0;">
          ${['Server-side SHA-256 hashing','Integrity verification','Authenticity analysis (EfficientNet)','Grad-CAM explainability','Optional face verification','Forensic PDF report'].map((s,i) => `
            <div class="progress-step"><div class="ps-icon">${i+1}</div><div class="ps-label">${s}</div></div>
          `).join('')}
        </div>
      </div>
    </div>
  `;
  return appShell('investigator', '#/upload', 'New Evidence', inner);
}

function postRenderUpload() {
  const dz = $('#invDropzone');
  const input = $('#invFileInput');
  dz.addEventListener('click', () => input.click());
  dz.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') input.click(); });
  ['dragenter','dragover'].forEach(evt => dz.addEventListener(evt, (e) => { e.preventDefault(); dz.classList.add('drag'); }));
  ['dragleave','drop'].forEach(evt => dz.addEventListener(evt, (e) => { e.preventDefault(); dz.classList.remove('drag'); }));
  dz.addEventListener('drop', (e) => { const f = e.dataTransfer.files[0]; if (f) handleFile(f); });
  input.addEventListener('change', (e) => { const f = e.target.files[0]; if (f) handleFile(f); });

  const createBtn = $('#createCaseBtn');
  if (createBtn) {
    createBtn.addEventListener('click', async () => {
      const title = ($('#newCaseTitle').value || '').trim();
      if (!title) { toast('Title required', 'Enter a case title first.', 'error'); return; }
      createBtn.disabled = true;
      try {
        const created = await MayaApi.cases.create({ title });
        toast('Case created', `${created.case_number} is ready for evidence.`, 'success');
        router(); // reload the page so the new case appears in the selector
      } catch (err) {
        handleApiError(err, 'Could not create case');
      } finally {
        createBtn.disabled = false;
      }
    });
  }

  async function handleFile(f) {
    const select = $('#uploadCase');
    if (!select || !select.value) {
      toast('No case selected', 'Create or select a case before uploading evidence.', 'error');
      return;
    }
    const ext = '.' + (f.name.split('.').pop() || '').toLowerCase();
    if (!ACCEPTED_UPLOAD_EXTS.includes(ext)) {
      toast('Unsupported file type',
        `Use an image or video (${ACCEPTED_UPLOAD_EXTS.join(', ')}).`, 'error', 6000);
      return;
    }

    const preview = $('#invFilePreview');
    preview.innerHTML = `<div class="divider"></div><p style="color:var(--text-muted);">Uploading and hashing ${escapeHtml(f.name)}…</p>`;

    let previewDataUrl = null;
    if (f.type.startsWith('image/')) {
      previewDataUrl = await new Promise(res => {
        const r = new FileReader(); r.onload = () => res(r.result); r.readAsDataURL(f);
      });
    }

    let record;
    try {
      // The backend computes and stores the authoritative SHA-256.
      record = await MayaApi.evidence.upload(select.value, f);
    } catch (err) {
      preview.innerHTML = '';
      handleApiError(err, 'Upload failed');
      return;
    }

    const ev = MayaApi.adapt.evidence(record);
    setCtx({ activeEvidenceId: ev.backendId, activePreview: previewDataUrl });
    toast('Evidence stored', `${ev.filename} uploaded and hashed server-side.`, 'success', 3000);

    preview.innerHTML = `
      <div class="divider"></div>
      <div class="file-card">
        <div class="file-thumb">${previewDataUrl ? `<img src="${previewDataUrl}" alt="">` : ic(fileIconFor(ev.type))}</div>
        <div class="file-meta">
          <div class="fname">${escapeHtml(ev.filename)}</div>
          <div class="file-meta-row">
            <span class="kv">Type <b>${ev.type}</b></span>
            <span class="kv">Size <b>${fmtBytes(ev.size)}</b></span>
            <span class="kv">Evidence ID <b>EV-${ev.backendId}</b></span>
          </div>
          <div class="file-meta-row">
            <span class="kv">Uploaded <b>${fmtDate(ev.timestamp)}</b></span>
            <span class="kv">Status <b>${escapeHtml(ev.evidenceStatus)}</b></span>
          </div>
          <div class="file-meta-row"><span class="kv">SHA-256 (server-computed)</span></div>
          <div class="hash-box" style="margin-top:6px;"><span>${escapeHtml(ev.sha256)}</span></div>
        </div>
      </div>
      <button class="btn btn-primary" id="analyzeBtn" style="margin-top:16px;">${ic('analysis')} Analyze Evidence</button>
    `;
    $('#analyzeBtn').addEventListener('click', () => {
      location.hash = '#/analysis?id=' + ev.backendId + '&new=1';
    });
  }
}

/* Analysis route: if ?new=1 present with matching context evidence -> run processing animation,
   else show result (or a picker if nothing in context / no id). */
/* Load an evidence record together with its most recent AnalysisRun. */
async function loadEvidenceWithAnalysis(evidenceId) {
  const raw = await MayaApi.evidence.get(evidenceId);
  const ev = MayaApi.adapt.evidence(raw);
  let runs = [];
  try {
    runs = await MayaApi.evidence.analyses(evidenceId);
  } catch (err) {
    if (!(err instanceof MayaApi.ApiError && err.isNotFound)) throw err;
  }
  const latest = isVideoEvidence(ev)
    ? runs.slice().sort((a, b) => (b.analysis_id || 0) - (a.analysis_id || 0))[0] || null
    : runs.find(r => r.analysis_status === 'COMPLETED') || runs[0] || null;
  MayaApi.adapt.applyAnalysis(ev, latest);
  return { ev, analysis: latest ? MayaApi.adapt.analysis(latest) : null, runs };
}

async function renderAnalysisRoute() {
  const params = window.__routeParams;
  const id = params.get('id');
  const isNew = params.get('new') === '1';

  if (!id) return renderAnalysisPicker();

  if (isNew) {
    const raw = await MayaApi.evidence.get(id);
    return renderAnalysisProcessing(MayaApi.adapt.evidence(raw));
  }

  const { ev, analysis } = await loadEvidenceWithAnalysis(id);
  if (!analysis) return renderAnalysisNotRun(ev);
  return renderAnalysisResult(ev, analysis);
}

/* Evidence exists but has no completed analysis — offer to run the real one
   rather than displaying a placeholder verdict. */
function renderAnalysisNotRun(ev) {
  window.__postRender = () => {
    const btn = $('#runAnalysisBtn');
    if (btn) btn.addEventListener('click', () => {
      location.hash = `#/analysis?id=${ev.backendId}&new=1`;
    });
  };
  const inner = `
    <p class="page-sub">Evidence ID: <span class="mono">EV-${ev.backendId}</span> — ${escapeHtml(ev.filename)}</p>
    <div class="card">
      <div class="panel-title"><h3>No analysis yet</h3><span class="sub">${escapeHtml(ev.analysisStatus)}</span></div>
      <p style="color:var(--text-muted);font-size:13.5px;">
        This evidence has been uploaded and hashed, but the MAYA authenticity model has not produced a result for it yet.
      </p>
      <div class="hash-box" style="margin:12px 0;"><span>${escapeHtml(ev.sha256)}</span></div>
      <button class="btn btn-primary" id="runAnalysisBtn">${ic('analysis')} Run authenticity analysis</button>
    </div>
  `;
  return appShell('investigator', '#/analysis', 'Analysis', inner);
}

/* Picker lists real cases, then evidence for the selected case — every
   forensic tool stays on its own route (no Workspace → Cases redirect). */
const FORENSIC_SELECTION = {
  analysis: { caseId: null },
  xai: { caseId: null },
  tampering: { caseId: null },
  custody: { caseId: null },
  integrity: { caseId: null },
  face: { caseId: null },
};

async function renderForensicCaseEvidencePicker({
  toolKey,
  navPath,
  title,
  subtitle,
  actionLabel,
  buildHref,
}) {
  const params = window.__routeParams;
  const caseParam = params.get('case');
  if (caseParam) FORENSIC_SELECTION[toolKey].caseId = caseParam;

  const cases = (await MayaApi.cases.list()).map(MayaApi.adapt.case)
    .filter(c => String(c.status || '').toLowerCase() !== 'closed');
  const selectedCaseId = FORENSIC_SELECTION[toolKey].caseId
    || (cases[0] ? String(cases[0].backendId) : null);

  if (selectedCaseId) FORENSIC_SELECTION[toolKey].caseId = String(selectedCaseId);

  let evidence = [];
  let selectedCase = null;
  if (selectedCaseId) {
    selectedCase = cases.find(c => String(c.backendId) === String(selectedCaseId)) || null;
    try {
      const items = await MayaApi.evidence.listByCase(selectedCaseId);
      evidence = items.map(raw => {
        const ev = MayaApi.adapt.evidence(raw, selectedCaseId);
        ev.caseNumber = selectedCase ? selectedCase.caseNumber : '';
        return ev;
      });
    } catch (err) {
      evidence = [];
    }
  }

  window.__postRender = () => {
    const sel = $('#forensicCaseSelect');
    if (sel) {
      sel.addEventListener('change', () => {
        FORENSIC_SELECTION[toolKey].caseId = sel.value;
        location.hash = `${navPath}?case=${encodeURIComponent(sel.value)}`;
      });
    }
  };

  const inner = `
    <p class="page-sub">${escapeHtml(subtitle)}</p>
    <div class="card" style="margin-bottom:16px;">
      <div class="panel-title"><h3>Select Case</h3><span class="sub">${cases.length} open</span></div>
      ${cases.length ? `
        <label for="forensicCaseSelect" style="font-size:12.5px;color:var(--text-muted);">Case</label>
        <select id="forensicCaseSelect" style="width:100%;margin-top:6px;padding:10px 12px;border-radius:10px;background:var(--panel-2);border:1px solid var(--border);color:var(--text);">
          ${cases.map(c => `<option value="${c.backendId}" ${String(c.backendId) === String(selectedCaseId) ? 'selected' : ''}>${escapeHtml(c.caseNumber)} — ${escapeHtml(c.name)}</option>`).join('')}
        </select>
      ` : `<p style="color:var(--text-muted);font-size:13.5px;margin:0;">No open cases yet. <a href="#/upload" style="color:var(--cyan);">Upload evidence</a> to create one.</p>`}
    </div>
    <div class="card">
      <div class="panel-title"><h3>Evidence in this case</h3>
        <a href="#/upload" class="btn btn-primary btn-sm">${ic('upload')} New Evidence</a>
      </div>
      <div class="table-wrap">
        <table class="data-table">
          <thead><tr><th>Evidence ID</th><th>Filename</th><th>Analysis</th><th>Uploaded</th><th></th></tr></thead>
          <tbody>${evidence.length ? evidence.map(e => `
            <tr>
              <td class="id-cell">EV-${e.backendId}</td>
              <td>${escapeHtml(e.filename)}</td>
              <td>${statusPill(e.analysisStatus)}</td>
              <td>${fmtDateShort(e.timestamp)}</td>
              <td><a href="${buildHref(e)}" class="btn btn-ghost btn-sm">${escapeHtml(actionLabel)}</a></td>
            </tr>`).join('')
            : `<tr><td colspan="5" style="color:var(--text-muted);">${selectedCaseId ? 'No evidence in this case yet.' : 'Select a case to continue.'}</td></tr>`}
          </tbody>
        </table>
      </div>
    </div>
  `;
  return appShell('investigator', navPath, title, inner);
}

/* Picker lists real evidence, gathered per case (ownership enforced by the
   backend on every call). */
async function renderAnalysisPicker() {
  return renderForensicCaseEvidencePicker({
    toolKey: 'analysis',
    navPath: '#/analysis',
    title: 'Analysis',
    subtitle: 'Select a case, then choose evidence to run or view authenticity analysis.',
    actionLabel: 'Open',
    buildHref: (e) => `#/analysis?id=${e.backendId}&case=${e.caseBackendId || FORENSIC_SELECTION.analysis.caseId || ''}`,
  });
}

/* The progress list mirrors the real backend pipeline stages; the request is
   genuinely in flight while it animates. */
function renderAnalysisProcessing(ev) {
  window.__postRender = () => postRenderAnalysisProcessing(ev);
  const videoPending = isVideoEvidence(ev) && window.MayaVideoResult
    ? MayaVideoResult.beginNewAnalysis()
    : '';
  const stepNames = isVideoEvidence(ev) && window.MayaVideoResult
    ? MayaVideoResult.videoProgressSteps()
    : ['Integrity verification (SHA-256)','Model inference (EfficientNet)','Grad-CAM explainability','Persisting investigation artifacts'];
  const inner = `
    ${videoPending}
    <div class="card" style="max-width:640px;margin:0 auto;">
      <div style="text-align:center;margin-bottom:6px;">
        <div class="fi" style="margin:0 auto 14px;width:52px;height:52px;">${ic('scan')}</div>
        <h2 style="font-size:18px;">Analyzing ${escapeHtml(ev.filename)}</h2>
        <p style="color:var(--text-muted);font-size:13px;margin-top:4px;">Evidence ID: <span class="mono">EV-${ev.backendId}</span></p>
      </div>
      <div class="progress-list" id="analysisProgressList" style="margin-top:26px;">
        ${stepNames.map((s,i) => `<div class="progress-step" data-i="${i}"><div class="ps-icon">${i+1}</div><div class="ps-label">${s}</div><div class="ps-bar"><span></span></div></div>`).join('')}
      </div>
      <p id="analysisProgressNote" style="color:var(--text-muted);font-size:12.5px;text-align:center;margin-top:18px;">
        Running on the MAYA backend — this can take a few seconds on CPU.
      </p>
    </div>
  `;
  return appShell('investigator', '#/upload', 'Analyzing Evidence', inner);
}

/* Kicks off the real POST /api/evidence/<id>/analyze and only advances the
   final step when the backend actually answers. Failures surface the real
   error instead of completing the animation. */
function postRenderAnalysisProcessing(ev) {
  if (window.__videoTemporalChart) {
    window.__videoTemporalChart.destroy();
    window.__videoTemporalChart = null;
  }
  window.__xaiExportData = null;
  const rows = $$('#analysisProgressList .progress-step');
  const note = $('#analysisProgressNote');
  let settled = false;
  let i = 0;

  const request = MayaApi.analysis.analyze(ev.backendId, {
    generateExplanation: true,
    explainer: 'gradcam',
    verifyBeforeAnalyze: true,
  });

  function markDone(row) {
    row.classList.remove('active');
    row.classList.add('done');
    row.querySelector('.ps-icon').innerHTML = ic('check');
  }

  function step() {
    // Hold on the last stage until the backend responds.
    if (i >= rows.length - 1) return;
    const row = rows[i];
    row.classList.add('active');
    const bar = row.querySelector('.ps-bar span');
    requestAnimationFrame(() => bar.style.width = '100%');
    setTimeout(() => { if (settled) return; markDone(row); i++; step(); }, 420);
  }
  step();

  const last = rows[rows.length - 1];
  last.classList.add('active');

  request.then((run) => {
    settled = true;
    rows.forEach(markDone);
    if (run.analysis_status === 'COMPLETED' && run.video_analysis) {
      const score = window.MayaVideoResult && MayaVideoResult.formatConfidenceScore(run.video_analysis);
      toast('Analysis complete', `${run.video_analysis.prediction || run.prediction}${score ? ' · Confidence Score ' + score : ''}.`, 'success');
    } else if (run.analysis_status === 'COMPLETED') {
      toast('Analysis complete', `${run.prediction} at ${Number(run.confidence).toFixed(1)}% confidence.`, 'success');
    } else {
      toast('Analysis finished', `Status: ${run.analysis_status}`, 'info');
    }
    setTimeout(() => { location.hash = '#/analysis?id=' + ev.backendId; }, 350);
  }).catch((err) => {
    settled = true;
    rows.forEach(r => { r.classList.remove('active'); });
    last.classList.add('failed');
    if (note) {
      note.innerHTML = `<span style="color:var(--red);">Analysis failed: ${escapeHtml(
        err instanceof MayaApi.ApiError ? `${err.message} (${err.status} ${err.code})` : String(err.message || err)
      )}</span>`;
    }
    handleApiError(err, 'Analysis failed');
  });
}

/* Every value on this page comes from the stored AnalysisRun. */
function renderAnalysisResult(ev, a) {
  if (isVideoEvidence(ev) && a.analysisStatus === 'FAILED') {
    const failure = window.MayaVideoResult
      ? MayaVideoResult.renderVideoFailure(a.errorMessage)
      : '<section class="video-analysis" data-video-state="failed"><div class="card"><p class="video-status-error">Analysis failed.</p></div></section>';
    return appShell('investigator', '#/analysis', 'Analysis Result', failure);
  }
  if (a.videoAnalysis && window.MayaVideoResult) {
    window.__postRender = () => postRenderVideoAnalysis(a);
    const sha = ev.sha256 ? ev.sha256.slice(0, 16) + '…' : '—';
    const inner = MayaVideoResult.renderVideoAnalysisSection(a, {
      analysisId: a.analysisId,
      artifactUrl: MayaApi.analysis.artifactUrl,
      evidenceLabel: ev.filename || ('EV-' + ev.backendId),
      sha256: sha,
    });
    return appShell('investigator', '#/analysis', 'Analysis Result', inner);
  }
  window.__postRender = () => postRenderAnalysisResult(ev, a);
  const tone = a.status === 'Authentic' ? 'green' : a.status === 'Tampered' ? 'red' : 'blue';
  const statusText = {
    Authentic: 'The model classified this image as REAL — no manipulation signature was detected.',
    Tampered: 'The model classified this image as FAKE — a manipulation signature was detected.',
  }[a.status] || 'The model did not return a conclusive prediction for this evidence.';
  const confidence = typeof a.confidence === 'number' ? a.confidence : null;
  const ring = confidence === 0 || confidence ? confidence : 0;

  const inner = `
    <p class="page-sub">Evidence ID: <span class="mono">EV-${ev.backendId}</span> · Investigation <span class="mono">${escapeHtml(a.investigationId || '—')}</span></p>

    <div class="status-banner" style="border-color:var(--${tone});background:var(--${tone}-dim);">
      <div class="sb-icon" style="background:rgba(0,0,0,0.2);color:var(--${tone});">${ic(a.status==='Authentic'?'checkCircle':a.status==='Tampered'?'xCircle':'alert')}</div>
      <div><h3 style="color:var(--${tone});">${escapeHtml(a.prediction || 'UNKNOWN')}</h3><p>${statusText}</p></div>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="panel-title"><h3>Model Confidence</h3><span class="sub">${escapeHtml(a.modelName || 'model')}</span></div>
        <div class="score-ring-wrap">
          <div class="score-ring">
            <svg viewBox="0 0 120 120">
              <circle class="bg" cx="60" cy="60" r="52"></circle>
              <circle class="fg" cx="60" cy="60" r="52" style="stroke:var(--${tone});stroke-dasharray:${2*Math.PI*52};stroke-dashoffset:${2*Math.PI*52};"></circle>
            </svg>
            <div class="val"><div class="n">${confidence !== null ? confidence.toFixed(1) + '%' : '—'}</div><div class="l">MODEL CONFIDENCE</div></div>
          </div>
        </div>
        <div class="kv-grid">
          <div class="kv-item"><div class="kl">REAL probability</div><div class="kv-val">${typeof a.realProbability === 'number' ? (a.realProbability * 100).toFixed(2) + '%' : '—'}</div></div>
          <div class="kv-item"><div class="kl">FAKE probability</div><div class="kv-val">${typeof a.fakeProbability === 'number' ? (a.fakeProbability * 100).toFixed(2) + '%' : '—'}</div></div>
          <div class="kv-item"><div class="kl">SHA-256</div><div class="kv-val">${(ev.sha256||'—').slice(0,20)}…</div></div>
          <div class="kv-item"><div class="kl">File type</div><div class="kv-val">${ev.type}</div></div>
          <div class="kv-item"><div class="kl">Size</div><div class="kv-val">${fmtBytes(ev.size)}</div></div>
          <div class="kv-item"><div class="kl">Model version</div><div class="kv-val">${escapeHtml(a.modelVersion || '—')}</div></div>
          <div class="kv-item"><div class="kl">Trust score</div><div class="kv-val">${a.trustScore !== null && a.trustScore !== undefined ? Number(a.trustScore).toFixed(2) : 'Not computed'}</div></div>
          <div class="kv-item"><div class="kl">Completed</div><div class="kv-val">${a.completedAt ? fmtDate(a.completedAt) : '—'}</div></div>
        </div>
        <p style="color:var(--text-muted);font-size:12px;margin:12px 0 0;">
          MAYA reports a binary REAL/FAKE classification with a confidence value. Metadata, compression and container-level forensics are not part of this model.
        </p>
      </div>

      <div style="display:flex;flex-direction:column;gap:18px;">
        <div class="overview-card card">
          <div class="oc-head"><div class="oc-icon" style="background:var(--cyan-dim);color:var(--cyan);">${ic('xai')}</div><h4>XAI Overview</h4></div>
          <p class="oc-text">${a.explanation && a.explanation.heatmap
            ? `Grad-CAM explanation available (${escapeHtml(a.explanation.explainer || 'gradcam')}).`
            : 'No explainability artifact was produced for this analysis.'}</p>
          <div style="display:flex;align-items:center;justify-content:space-between;">
            <span class="pill pill-cyan">Confidence: ${confidence !== null ? confidence.toFixed(1) + '%' : '—'}</span>
            <a href="#/xai?id=${ev.backendId}" class="btn btn-ghost btn-sm">View Details ${ic('chevronRight')}</a>
          </div>
        </div>
        <div class="overview-card card">
          <div class="oc-head"><div class="oc-icon" style="background:var(--red-dim);color:var(--red);">${ic('tampering')}</div><h4>Attention Heat Map</h4></div>
          <p class="oc-text">${a.explanation && (a.explanation.heatmap || a.explanation.overlay)
            ? 'Grad-CAM highlights the pixels that most influenced the verdict.'
            : 'No heat map artifact is available for this analysis.'}</p>
          <div style="display:flex;align-items:center;justify-content:space-between;">
            <span class="pill pill-${tone}">${escapeHtml(a.prediction || '—')}</span>
            <a href="#/tampering?id=${ev.backendId}" class="btn btn-ghost btn-sm">View Heat Map ${ic('chevronRight')}</a>
          </div>
        </div>
        <div class="card" style="display:flex;gap:10px;flex-wrap:wrap;">
          <a href="#/chain-of-custody?id=${ev.backendId}" class="btn btn-ghost btn-sm">${ic('custody')} Chain of Custody</a>
          <a href="#/integrity?id=${ev.backendId}" class="btn btn-ghost btn-sm">${ic('security')} Integrity</a>
          <a href="#/face-verification?id=${ev.backendId}" class="btn btn-ghost btn-sm">${ic('investigators')} Face Verification</a>
          <button class="btn btn-primary btn-sm" data-action="generate-report" data-analysis="${a.analysisId}">${ic('reports')} Generate Report</button>
          <button class="btn btn-ghost btn-sm" data-action="verify-integrity" data-evidence="${ev.backendId}">${ic('security')} Re-verify Integrity</button>
        </div>
      </div>
    </div>
    ${isVideoEvidence(ev) && (a.frames || []).length ? videoFrameGalleryHtml(ev, a, defaultVideoFrameNumber(a), { links: true }) : ''}
  `;
  return appShell('investigator', '#/analysis', 'Analysis Result', inner);
}

function postRenderVideoAnalysis(a) {
  if (window.__videoTemporalChart) {
    window.__videoTemporalChart.destroy();
    window.__videoTemporalChart = null;
  }
  const canvas = document.getElementById('videoTemporalChart');
  const points = window.MayaVideoResult ? MayaVideoResult.temporalPoints(a.videoAnalysis) : [];
  if (canvas && points.length && window.Chart) {
    window.__videoTemporalChart = new Chart(canvas, {
      type: 'bar',
      data: {
        labels: points.map((point) => String(point.frameIndex)),
        datasets: [{
          label: 'Relative temporal contribution',
          data: points.map((point) => point.contribution),
          backgroundColor: points.map((point) => point.fallback ? '#f5a623' : '#29e0d6'),
        }],
      },
      options: {
        responsive: true,
        plugins: { legend: { labels: { color: '#8ca0b3' } } },
        scales: {
          x: { ticks: { color: '#8ca0b3' }, title: { display: true, text: 'Frame index', color: '#8ca0b3' } },
          y: { ticks: { color: '#8ca0b3' }, title: { display: true, text: 'Relative contribution', color: '#8ca0b3' }, beginAtZero: true },
        },
      },
    });
  }
  document.querySelectorAll('img.video-gradcam, img.video-contact-sheet').forEach((img) => {
    img.addEventListener('error', () => {
      const note = document.createElement('p');
      note.className = 'video-missing-artifact';
      note.textContent = 'Grad-CAM artifact is not available.';
      img.replaceWith(note);
    });
  });
}

function postRenderAnalysisResult(ev, a) {
  if (window.__videoTemporalChart) {
    window.__videoTemporalChart.destroy();
    window.__videoTemporalChart = null;
  }
  const fg = $('.score-ring .fg');
  if (!fg) return;
  const pct = typeof a.confidence === 'number' ? a.confidence : 0;
  requestAnimationFrame(() => { fg.style.strokeDashoffset = (2*Math.PI*52) * (1 - pct/100); });
}

/* ==========================================================================
   13. XAI DETAILS PAGE
   ========================================================================== */
/* Advanced-XAI results are optional: the backend records them only when the
   caller opted in, and a failure there is non-fatal by design. */
function advancedXaiRows(adv) {
  if (!adv || typeof adv !== 'object') return [];
  const rows = [];
  const push = (label, value) => {
    if (value === null || value === undefined) return;
    rows.push([label, typeof value === 'number' ? value.toFixed(3) : String(value)]);
  };
  push('Trust score', adv.trust_score);
  push('Quality score', adv.quality_score);
  if (Array.isArray(adv.methods_run) && adv.methods_run.length) {
    rows.push(['Methods run', adv.methods_run.join(', ')]);
  }
  Object.keys(adv).forEach(k => {
    if (['trust_score', 'quality_score', 'methods_run', 'errors'].includes(k)) return;
    const v = adv[k];
    if (typeof v === 'number' || typeof v === 'string' || typeof v === 'boolean') {
      rows.push([k.replace(/_/g, ' '), String(v)]);
    }
  });
  return rows;
}

async function renderXAI() {
  const params = window.__routeParams;
  const id = params.get('id');
  if (!id) {
    return renderForensicCaseEvidencePicker({
      toolKey: 'xai',
      navPath: '#/xai',
      title: 'XAI Insights',
      subtitle: 'Select a case and evidence to view Grad-CAM explainability for its analysis.',
      actionLabel: 'Open XAI',
      buildHref: (e) => `#/xai?id=${e.backendId}&case=${e.caseBackendId || FORENSIC_SELECTION.xai.caseId || ''}`,
    });
  }

  const { ev, analysis: a } = await loadEvidenceWithAnalysis(id);
  if (!a) {
    return appShell('investigator', '#/xai', 'XAI Insights', `
      <div class="empty-state card">${ic('xai')}
        <p style="margin-top:10px;">No completed analysis for EV-${escapeHtml(String(id))} yet.</p>
        <a href="#/analysis?id=${escapeHtml(String(id))}" class="btn btn-primary btn-sm" style="margin-top:12px;">Run analysis</a>
        <a href="#/xai" class="btn btn-ghost btn-sm" style="margin-top:12px;">Back to case picker</a>
      </div>`);
  }
  if (a.videoAnalysis && window.MayaVideoResult) {
    window.__postRender = () => {
      postRenderVideoAnalysis(a);
      postRenderExpanders();
    };
    window.__xaiExportData = { ev, a };
    const inner = `
      <div class="breadcrumb"><a href="#/analysis?id=${ev.backendId}">Analysis</a> ${ic('chevronRight')} <span>XAI Insights</span></div>
      <div class="page-sub" style="margin-top:-8px;">Evidence ID: <span class="mono">EV-${ev.backendId}</span> — ${escapeHtml(ev.filename)}</div>
      ${MayaVideoResult.renderVideoExplainability(a, {
        analysisId: a.analysisId,
        artifactUrl: MayaApi.analysis.artifactUrl,
      })}
      <button class="btn btn-primary" id="downloadXaiBtn">${ic('download')} Download XAI Summary</button>
    `;
    return appShell('investigator', '#/xai', 'XAI Insights', inner);
  }
  window.__postRender = postRenderExpanders;

  const confidence = typeof a.confidence === 'number' ? a.confidence : null;
  const hasHeatmap = !!(a.explanation && a.explanation.heatmap);
  const hasOverlay = !!(a.explanation && a.explanation.overlay);
  const advRows = advancedXaiRows(a.advancedXai);

  const tone = a.status === 'Authentic' ? 'green' : a.status === 'Tampered' ? 'red' : 'blue';

  const inner = `
    <div class="breadcrumb"><a href="#/analysis?id=${ev.backendId}">Analysis</a> ${ic('chevronRight')} <span>Explainable AI Analysis</span></div>
    <div class="page-sub" style="margin-top:-8px;">Evidence ID: <span class="mono">EV-${ev.backendId}</span> — ${escapeHtml(ev.filename)}</div>

    <div class="grid-2">
      <div style="display:flex;flex-direction:column;gap:18px;">
        <div class="card">
          <div class="panel-title"><h3>AI Prediction</h3><span class="sub">${escapeHtml(a.modelName || '')}</span></div>
          <div class="status-banner" style="border-color:var(--${tone});background:var(--${tone}-dim);margin-bottom:0;">
            <div class="sb-icon" style="background:rgba(0,0,0,0.2);color:inherit;">${ic('xai')}</div>
            <div><h3>${escapeHtml(a.prediction || 'UNKNOWN')}</h3><p>Model confidence: ${confidence !== null ? confidence.toFixed(1) + '%' : 'not reported'}</p></div>
          </div>
        </div>

        <div class="card">
          <div class="panel-title"><h3>Grad-CAM Attribution</h3><span class="sub">${escapeHtml((a.explanation && a.explanation.explainer) || 'none')}</span></div>
          ${hasOverlay || hasHeatmap ? `
            <img src="${hasOverlay ? MayaApi.analysis.artifactUrl(a.analysisId, 'overlay') : MayaApi.analysis.artifactUrl(a.analysisId, 'heatmap')}"
                 alt="Grad-CAM attribution" style="width:100%;border-radius:10px;display:block;"
                 onerror="this.replaceWith(Object.assign(document.createElement('p'),{textContent:'Artifact image could not be loaded from the backend.',style:'color:var(--text-muted);font-size:13px;'}))">
            <p style="color:var(--text-muted);font-size:12.5px;margin:10px 0 0;">
              Warmer areas contributed most to the verdict. This is model attribution, not a proof of manipulation.
            </p>
            <a href="#/tampering?id=${ev.backendId}" class="btn btn-ghost btn-sm" style="margin-top:12px;">${ic('tampering')} Compare with original</a>
          ` : `<p style="color:var(--text-muted);font-size:13.5px;margin:0;">
              No Grad-CAM artifact was produced for this analysis. Explainability is optional and a failure there does not invalidate the verdict above.
            </p>`}
        </div>

        <div class="card">
          <div class="panel-title"><h3>Advanced XAI</h3><span class="sub">${advRows.length ? 'opt-in results' : 'not requested'}</span></div>
          ${advRows.length ? `<div class="kv-grid">${advRows.map(([k, v]) => `
              <div class="kv-item"><div class="kl">${escapeHtml(k)}</div><div class="kv-val">${escapeHtml(v)}</div></div>`).join('')}</div>`
            : `<p style="color:var(--text-muted);font-size:13.5px;margin:0;">
              SHAP, faithfulness, counterfactual and fusion metrics are expensive and only run when explicitly requested for an analysis. None were recorded for this run.
            </p>`}
        </div>
      </div>

      <div style="display:flex;flex-direction:column;gap:12px;">
        <div class="card">
          <div class="panel-title"><h3>Model Provenance</h3></div>
          <div class="kv-grid">
            <div class="kv-item"><div class="kl">Investigation</div><div class="kv-val">${escapeHtml(a.investigationId || '—')}</div></div>
            <div class="kv-item"><div class="kl">Model</div><div class="kv-val">${escapeHtml(a.modelName || '—')}</div></div>
            <div class="kv-item"><div class="kl">Model version</div><div class="kv-val">${escapeHtml(a.modelVersion || '—')}</div></div>
            <div class="kv-item"><div class="kl">Dataset version</div><div class="kv-val">${escapeHtml(a.datasetVersion || '—')}</div></div>
            <div class="kv-item"><div class="kl">Trust score</div><div class="kv-val">${a.trustScore !== null && a.trustScore !== undefined ? Number(a.trustScore).toFixed(3) : 'Not computed'}</div></div>
            <div class="kv-item"><div class="kl">Quality score</div><div class="kv-val">${a.qualityScore !== null && a.qualityScore !== undefined ? Number(a.qualityScore).toFixed(3) : 'Not computed'}</div></div>
          </div>
        </div>

        <button class="btn btn-primary" id="downloadXaiBtn" style="margin-top:6px;">${ic('download')} Download XAI Summary</button>
        <button class="btn btn-ghost" data-action="generate-report" data-analysis="${a.analysisId}">${ic('reports')} Generate Forensic PDF</button>
      </div>
    </div>
  `;
  window.__xaiExportData = { ev, a };
  return appShell('investigator', '#/xai', 'Explainable AI Analysis', inner);
}

function analysisEmptyState() {
  return `<div class="empty-state card">${ic('analysis')}<p style="margin-top:10px;">Select a case and evidence from this forensic tool to continue.</p></div>`;
}

/* ==========================================================================
   INTEGRITY (self-contained forensic tool)
   ========================================================================== */
async function renderIntegrityRoute() {
  const params = window.__routeParams;
  const id = params.get('id');
  if (!id) {
    return renderForensicCaseEvidencePicker({
      toolKey: 'integrity',
      navPath: '#/integrity',
      title: 'Integrity',
      subtitle: 'Select a case and evidence to view or re-verify SHA-256 integrity.',
      actionLabel: 'Open Integrity',
      buildHref: (e) => `#/integrity?id=${e.backendId}&case=${e.caseBackendId || FORENSIC_SELECTION.integrity.caseId || ''}`,
    });
  }

  const raw = await MayaApi.evidence.get(id);
  const ev = MayaApi.adapt.evidence(raw);
  window.__postRender = () => {
    const btn = $('#integrityVerifyBtn');
    if (!btn) return;
    btn.addEventListener('click', async () => {
      btn.disabled = true;
      btn.textContent = 'Verifying…';
      try {
        const result = await MayaApi.evidence.verifyIntegrity(ev.backendId);
        const status = result.integrity_status || result.status || 'UNKNOWN';
        const match = result.match !== false && status === 'VALID';
        $('#integrityResultBox').innerHTML = `
          <div class="kv-grid">
            <div class="kv-item"><div class="kl">Integrity status</div><div class="kv-val" style="color:var(--${match ? 'green' : 'red'});">${escapeHtml(String(status))}</div></div>
            <div class="kv-item"><div class="kl">Stored SHA-256</div><div class="kv-val mono" style="font-size:11px;word-break:break-all;">${escapeHtml(result.stored_sha256 || ev.sha256 || '—')}</div></div>
            <div class="kv-item"><div class="kl">Current SHA-256</div><div class="kv-val mono" style="font-size:11px;word-break:break-all;">${escapeHtml(result.current_sha256 || '—')}</div></div>
          </div>`;
        toast(match ? 'Integrity VALID' : 'Integrity issue', `Status: ${status}`, match ? 'success' : 'error');
      } catch (err) {
        toast('Verification failed', err.message || 'Request failed', 'error', 6000);
      } finally {
        btn.disabled = false;
        btn.innerHTML = `${ic('security')} Re-verify Integrity`;
      }
    });
  };

  const inner = `
    <div class="breadcrumb"><a href="#/integrity">Integrity</a> ${ic('chevronRight')} <span>EV-${ev.backendId}</span></div>
    <p class="page-sub">${escapeHtml(ev.filename)} — SHA-256 recorded at upload</p>
    <div class="card">
      <div class="kv-grid">
        <div class="kv-item"><div class="kl">Evidence ID</div><div class="kv-val">EV-${ev.backendId}</div></div>
        <div class="kv-item"><div class="kl">Case ID</div><div class="kv-val">${ev.caseBackendId || '—'}</div></div>
        <div class="kv-item"><div class="kl">MIME type</div><div class="kv-val">${escapeHtml(ev.mimeType || '—')}</div></div>
        <div class="kv-item"><div class="kl">Size</div><div class="kv-val">${fmtBytes(ev.size)}</div></div>
        <div class="kv-item"><div class="kl">Evidence status</div><div class="kv-val">${escapeHtml(ev.evidenceStatus || '—')}</div></div>
        <div class="kv-item"><div class="kl">Uploaded</div><div class="kv-val">${fmtDate(ev.timestamp)}</div></div>
      </div>
      <div class="panel-title" style="margin-top:18px;"><h3>SHA-256 (at ingest)</h3></div>
      <div class="hash-box"><span>${escapeHtml(ev.sha256)}</span></div>
      <div style="margin-top:16px;display:flex;gap:10px;flex-wrap:wrap;">
        <button class="btn btn-primary btn-sm" id="integrityVerifyBtn">${ic('security')} Re-verify Integrity</button>
        <a href="#/integrity" class="btn btn-ghost btn-sm">Back to case picker</a>
      </div>
      <div id="integrityResultBox" style="margin-top:18px;"></div>
    </div>
  `;
  return appShell('investigator', '#/integrity', 'Integrity', inner);
}

/* ==========================================================================
   FACE VERIFICATION (Phase 5 endpoint — self-contained)
   ========================================================================== */
/* Plain-language explanation of every reason_code the Phase 5 pipeline can
   return (ai/face_verification/types.py). Nothing here changes the decision —
   it only makes the backend's own reason readable. */
const FACE_REASON_TEXT = {
  OK: 'Both images yielded exactly one usable face; the decision came from the similarity score against the threshold.',
  INCONCLUSIVE_SCORE: 'Similarity fell between the no-match and match thresholds, so the service refused to force a decision.',
  NO_FACE_REFERENCE: 'No face was detected in the uploaded reference image.',
  NO_FACE_EVIDENCE: 'No face was detected in the evidence image.',
  MULTIPLE_FACES_REFERENCE: 'More than one face was detected in the reference image, so the service could not tell which identity to compare.',
  MULTIPLE_FACES_EVIDENCE: 'More than one face was detected in the evidence image, so the service could not tell which identity to compare.',
  LOW_QUALITY_REFERENCE: 'The reference face failed the quality gate (size, sharpness or brightness).',
  LOW_QUALITY_EVIDENCE: 'The evidence face failed the quality gate (size, sharpness or brightness).',
  UNUSABLE_REFERENCE: 'The reference face crop was empty or unusable.',
  UNUSABLE_EVIDENCE: 'The evidence face crop was empty or unusable.',
  EMBEDDING_FAILURE: 'Face detection or embedding extraction raised an error; no scores were produced.',
  UNSUPPORTED_IMAGE: 'The image format is not supported by the verification service.',
  CORRUPTED_IMAGE: 'The image could not be decoded.',
};

/* Renders every field the backend actually returned. A field that the service
   genuinely did not produce is labelled as such instead of being hidden — an
   INCONCLUSIVE result still shows thresholds, face counts, engine and model. */
function faceVerificationResultHtml(row) {
  const NA = 'Not available — not produced by the verification service.';
  const num = (v, digits) => (v === null || v === undefined || v === '' ? null : Number(v).toFixed(digits));
  const decision = row.decision || null;
  const reasonCode = row.reason_code || null;
  const sim = num(row.similarity_score, 4);
  const dist = num(row.distance_score, 4);
  const match = num(row.threshold, 2);
  const noMatch = num(row.no_match_threshold, 2);
  const refFaces = row.reference_face_count;
  const evdFaces = row.evidence_face_count;
  const tone = decision === 'MATCH' ? 'green' : decision === 'NO_MATCH' ? 'red' : 'amber';

  const cell = (label, value) => `
    <div class="kv-item"><div class="kl">${escapeHtml(label)}</div>
      <div class="kv-val"${value === null ? ' style="color:var(--text-muted);font-size:12px;"' : ''}>${escapeHtml(value === null ? NA : String(value))}</div></div>`;

  const faceDetected = (refFaces === null || refFaces === undefined || evdFaces === null || evdFaces === undefined)
    ? null
    : (refFaces > 0 && evdFaces > 0 ? 'Yes — face found in both images' : 'No — at least one image had no detected face');

  return `
    <div class="status-banner" style="border-color:var(--${tone});background:var(--${tone}-dim);">
      <div class="sb-icon" style="background:rgba(0,0,0,0.2);color:inherit;">${ic('investigators')}</div>
      <div>
        <h3>${escapeHtml(decision || 'NO DECISION')}</h3>
        <p>${escapeHtml(reasonCode ? (FACE_REASON_TEXT[reasonCode] || `Reason code: ${reasonCode}`) : NA)}</p>
      </div>
    </div>
    <div class="kv-grid" style="margin-top:14px;">
      ${cell('Result', decision)}
      ${cell('Verification status', row.verification_status || null)}
      ${cell('Similarity score (cosine)', sim)}
      ${cell('Distance (euclidean)', dist)}
      ${cell('Match threshold', match)}
      ${cell('No-match threshold', noMatch)}
      ${cell('Reason code', reasonCode)}
      ${cell('Face detected', faceDetected)}
      ${cell('Faces in reference image', refFaces === null || refFaces === undefined ? null : refFaces)}
      ${cell('Faces in evidence image', evdFaces === null || evdFaces === undefined ? null : evdFaces)}
      ${cell('Reference filename', row.reference_filename || null)}
      ${cell('Evidence ID', row.evidence_id === null || row.evidence_id === undefined ? null : `EV-${row.evidence_id}`)}
      ${cell('Verification ID', row.verification_id === null || row.verification_id === undefined ? null : row.verification_id)}
      ${cell('Investigation ID', row.investigation_id || null)}
      ${cell('Engine', row.engine_name || null)}
      ${cell('Model', row.model_name || null)}
      ${cell('Model version', row.model_version || null)}
      ${cell('Completed at', row.completed_at ? fmtDate(row.completed_at) : null)}
      ${row.error_message ? cell('Error', row.error_message) : ''}
    </div>
    ${sim === null ? `<p style="color:var(--text-muted);font-size:12.5px;margin:12px 0 0;">
      No similarity or distance was computed: the service stops before embedding comparison when a face cannot be
      isolated in both images, so these two fields do not exist for this run rather than being hidden.
    </p>` : ''}
    ${row.engine_name === 'opencv_haar' ? `<p style="color:var(--amber);font-size:12.5px;margin:10px 0 0;">
      Running on the OpenCV Haar fallback engine. Its identity discrimination is weaker than FaceNet, so treat
      borderline scores as indicative only.
    </p>` : ''}`;
}

async function renderFaceVerificationRoute() {
  const params = window.__routeParams;
  const id = params.get('id');
  if (!id) {
    return renderForensicCaseEvidencePicker({
      toolKey: 'face',
      navPath: '#/face-verification',
      title: 'Face Verification',
      subtitle: 'Select a case and evidence, then upload a reference face image for Phase 5 verification.',
      actionLabel: 'Verify Face',
      buildHref: (e) => `#/face-verification?id=${e.backendId}&case=${e.caseBackendId || FORENSIC_SELECTION.face.caseId || ''}`,
    });
  }

  const raw = await MayaApi.evidence.get(id);
  const ev = MayaApi.adapt.evidence(raw);
  let investigationId = '';
  try {
    const { analysis } = await loadEvidenceWithAnalysis(id);
    investigationId = (analysis && analysis.investigationId) || '';
  } catch (_) { /* optional */ }

  window.__postRender = () => {
    const input = $('#faceRefInput');
    const btn = $('#faceVerifyBtn');
    const out = $('#faceVerifyResult');
    if (!btn || !input) return;
    btn.addEventListener('click', async () => {
      const file = input.files && input.files[0];
      if (!file) {
        toast('Reference image required', 'Choose a reference face image before verifying.', 'error');
        return;
      }
      btn.disabled = true;
      btn.textContent = 'Verifying…';
      out.innerHTML = `<p style="color:var(--text-muted);font-size:13px;">Calling the face-verification endpoint…</p>`;
      try {
        const row = await MayaApi.faceVerification.create(ev.backendId, file, {
          investigationId: investigationId || undefined,
        });
        out.innerHTML = faceVerificationResultHtml(row);
        toast('Face verification complete', String(row.decision || row.verification_status || ''), 'success');
      } catch (err) {
        out.innerHTML = `<p style="color:var(--red);font-size:13.5px;">${escapeHtml(err.message || 'Verification failed')}</p>`;
        toast('Face verification failed', err.message || 'Request failed', 'error', 6000);
      } finally {
        btn.disabled = false;
        btn.innerHTML = `${ic('investigators')} Verify against reference`;
      }
    });
  };

  const inner = `
    <div class="breadcrumb"><a href="#/face-verification">Face Verification</a> ${ic('chevronRight')} <span>EV-${ev.backendId}</span></div>
    <p class="page-sub">Evidence: ${escapeHtml(ev.filename)} — uses POST /api/evidence/&lt;id&gt;/face-verification</p>
    <div class="card">
      <div class="kv-grid">
        <div class="kv-item"><div class="kl">Evidence ID</div><div class="kv-val">EV-${ev.backendId}</div></div>
        <div class="kv-item"><div class="kl">Investigation</div><div class="kv-val">${escapeHtml(investigationId || '—')}</div></div>
        <div class="kv-item"><div class="kl">MIME type</div><div class="kv-val">${escapeHtml(ev.mimeType || '—')}</div></div>
      </div>
      <div class="panel-title" style="margin-top:18px;"><h3>Reference face image</h3></div>
      <p style="color:var(--text-muted);font-size:13px;margin:0 0 10px;">Upload a reference image. It is sent to the backend as multipart field <code>file</code> and is not stored in localStorage.</p>
      <input type="file" id="faceRefInput" accept="image/*" style="margin-bottom:12px;">
      <div style="display:flex;gap:10px;flex-wrap:wrap;">
        <button class="btn btn-primary btn-sm" id="faceVerifyBtn">${ic('investigators')} Verify against reference</button>
        <a href="#/face-verification" class="btn btn-ghost btn-sm">Back to case picker</a>
      </div>
      <div id="faceVerifyResult" style="margin-top:18px;"></div>
    </div>
  `;
  return appShell('investigator', '#/face-verification', 'Face Verification', inner);
}

function postRenderExpanders() {
  $$('.expand-head').forEach(h => h.addEventListener('click', () => h.closest('.expand-item').classList.toggle('open')));
  const dl = $('#downloadXaiBtn');
  if (dl) dl.addEventListener('click', () => {
    const { ev, a } = window.__xaiExportData;
    // Text summary of real recorded values only. The authoritative artefact is
    // the backend-generated forensic PDF.
    const txt = [
      'MAYA / EVIDEX — XAI SUMMARY',
      '===========================',
      `Exported: ${fmtDate(new Date())}`,
      '',
      'EVIDENCE',
      '--------',
      `Evidence ID: EV-${ev.backendId}`,
      `Filename: ${ev.filename}`,
      `SHA-256 (server-computed): ${ev.sha256}`,
      `Size: ${ev.size} bytes`,
      `Uploaded: ${ev.timestamp}`,
      '',
      'ANALYSIS',
      '--------',
      `Investigation ID: ${a.investigationId || '—'}`,
      `Analysis ID: ${a.analysisId}`,
      `Prediction: ${(a.videoAnalysis && a.videoAnalysis.prediction) || a.prediction || '—'}`,
      a.videoAnalysis
        ? `Confidence Score: ${(window.MayaVideoResult && MayaVideoResult.formatConfidenceScore(a.videoAnalysis)) || 'not reported'}`
        : `Model confidence: ${typeof a.confidence === 'number' ? a.confidence.toFixed(2) + '%' : 'not reported'}`,
      `Model: ${(a.videoAnalysis && a.videoAnalysis.modelName) || a.modelName || '—'} (${(a.videoAnalysis && a.videoAnalysis.modelVersion) || a.modelVersion || '—'})`,
      `Dataset version: ${a.datasetVersion || '—'}`,
      `Trust score: ${a.trustScore ?? 'not computed'}`,
      `Quality score: ${a.qualityScore ?? 'not computed'}`,
      `Completed: ${a.completedAt || '—'}`,
      '',
      'EXPLAINABILITY',
      '--------------',
      a.videoAnalysis
        ? ((a.videoAnalysis.xaiAvailable === true && (a.videoAnalysis.gradcamContactSheet || (a.videoAnalysis.gradcamFrameIndices || []).length))
          ? 'Grad-CAM artifact available for this video analysis.'
          : 'Grad-CAM artifact is not available for this analysis.')
        : (a.explanation && (a.explanation.heatmap || a.explanation.overlay)
          ? `Grad-CAM artifact produced by "${a.explanation.explainer || 'gradcam'}".`
          : 'No Grad-CAM artifact was produced for this analysis.'),
      a.advancedXai ? `Advanced XAI recorded: ${JSON.stringify(a.advancedXai)}` : 'Advanced XAI: not requested.',
      '',
      'NOT AVAILABLE',
      '-------------',
      'Metadata, compression, noise-pattern, pixel-level, AI-generator and',
      'region-coordinate forensics are not implemented in MAYA and are',
      'deliberately omitted rather than estimated.',
      '',
      'MAYA reports a probabilistic REAL/FAKE classification. It is not a legal',
      'determination of authenticity or admissibility.',
    ].join('\n');
    downloadTextFile(`XAI_Summary_EV-${ev.backendId}.txt`, txt);
    toast('Download started', `XAI_Summary_EV-${ev.backendId}.txt`, 'success');
  });
}

function isVideoEvidence(ev) {
  return ev.type === 'Video'
    || ev.mediaType === 'video'
    || String(ev.mimeType || '').toLowerCase().startsWith('video/');
}

function explainedFrameNumbers(a) {
  const expl = a.explanation || {};
  const nums = [];
  const push = (n) => {
    const v = Number(n);
    if (Number.isInteger(v) && !nums.includes(v)) nums.push(v);
  };
  push(expl.frameNumber);
  (expl.explainedFrames || []).forEach(push);
  const heat = expl.heatmap || '';
  const match = String(heat).match(/frame_(\d+)/i);
  if (match) push(parseInt(match[1], 10));
  return nums;
}

function defaultVideoFrameNumber(a) {
  const expl = a.explanation || {};
  const primary = expl.frameNumber;
  if (primary !== null && primary !== undefined && Number.isInteger(Number(primary))) {
    return Number(primary);
  }
  const explained = explainedFrameNumbers(a);
  if (explained.length) return explained[0];
  const frames = a.frames || [];
  if (frames.length && Number.isInteger(Number(frames[0].frameNumber))) {
    return Number(frames[0].frameNumber);
  }
  return null;
}

function formatFrameTime(seconds) {
  if (typeof seconds !== 'number' || !Number.isFinite(seconds)) return '—';
  return seconds.toFixed(1) + 's';
}

function frameConfidenceLabel(frame) {
  if (!frame || typeof frame.confidence !== 'number') return '—';
  return frame.confidence.toFixed(1) + '%';
}

function videoOriginalUrl(ev, a, frameNumber) {
  if (isVideoEvidence(ev)) {
    if (frameNumber !== null && frameNumber !== undefined && frameNumber !== '') {
      return MayaApi.evidence.frameUrl(ev.backendId, frameNumber);
    }
    return MayaApi.analysis.artifactUrl(a.analysisId, 'original');
  }
  return MayaApi.evidence.fileUrl(ev.backendId);
}

function videoFrameGalleryHtml(ev, a, selectedFrame, opts) {
  const frames = a.frames || [];
  if (!frames.length) return '';
  const xai = new Set(explainedFrameNumbers(a));
  const links = !!(opts && opts.links);
  const tiles = frames.map((frame) => {
    const n = frame.frameNumber;
    const pred = frame.prediction || '—';
    const fake = pred === 'FAKE';
    const hasXai = xai.has(Number(n));
    const selected = Number(selectedFrame) === Number(n);
    const classes = [
      'frame-tile',
      fake ? 'is-fake' : '',
      hasXai ? 'has-xai' : '',
      selected ? 'selected' : '',
    ].filter(Boolean).join(' ');
    const inner = `
      <img src="${MayaApi.evidence.frameUrl(ev.backendId, n)}" alt="Frame ${n}" loading="lazy">
      <div class="ft-top">
        <span class="ft-frame">Frame ${n}</span>
        <span class="ft-badges">
          ${hasXai ? '<span class="ft-badge xai">XAI</span>' : ''}
          ${fake ? '<span class="ft-badge fake">FAKE</span>' : ''}
        </span>
      </div>
      <div class="ft-meta">
        <span>${formatFrameTime(frame.timestampSeconds)}</span>
        <span>${escapeHtml(pred)} · ${frameConfidenceLabel(frame)}</span>
      </div>`;
    if (links) {
      return `<a class="${classes}" href="#/tampering?id=${ev.backendId}&frame=${n}">${inner}</a>`;
    }
    return `<button type="button" class="${classes}" data-frame="${n}">${inner}</button>`;
  }).join('');
  return `
    <div class="card frame-gallery-card">
      <div class="panel-title"><h3>Analyzed frames</h3>
        <span class="sub">${frames.length} sampled · Grad-CAM on ${xai.size} frame${xai.size === 1 ? '' : 's'}</span>
      </div>
      <p class="page-sub" style="margin:0 0 12px;">
        Every sampled frame is listed. Grad-CAM is only produced for up to 3 selected frames; other tiles show the JPEG and prediction only.
      </p>
      <div class="frame-gallery">${tiles}</div>
    </div>`;
}

/* ==========================================================================
   14. TAMPERING HEAT MAP PAGE
   ========================================================================== */
/* Shows the real stored evidence image alongside the real Grad-CAM artifacts.
   MAYA does not emit region coordinates or per-region severities, so that
   section reports the gap instead of drawing invented boxes. */
async function renderTampering() {
  const params = window.__routeParams;
  const id = params.get('id');
  if (!id) {
    return renderForensicCaseEvidencePicker({
      toolKey: 'tampering',
      navPath: '#/tampering',
      title: 'Tampering Map',
      subtitle: 'Select a case and evidence to view the Grad-CAM attention heat map.',
      actionLabel: 'Open Map',
      buildHref: (e) => `#/tampering?id=${e.backendId}&case=${e.caseBackendId || FORENSIC_SELECTION.tampering.caseId || ''}`,
    });
  }

  const { ev, analysis: a } = await loadEvidenceWithAnalysis(id);
  if (!a) {
    return appShell('investigator', '#/tampering', 'Attention Heat Map', `
      <div class="empty-state card">${ic('tampering')}
        <p style="margin-top:10px;">No completed analysis for EV-${escapeHtml(String(id))} yet.</p>
        <a href="#/analysis?id=${escapeHtml(String(id))}" class="btn btn-primary btn-sm" style="margin-top:12px;">Run analysis</a>
        <a href="#/tampering" class="btn btn-ghost btn-sm" style="margin-top:12px;">Back to case picker</a>
      </div>`);
  }
  if (a.videoAnalysis && window.MayaVideoResult) {
    const requestedFrame = params.get('frame');
    window.__postRender = () => {
      postRenderVideoAnalysis(a);
      MayaVideoResult.bindTamperingMap(document);
    };
    const inner = `
      <div class="breadcrumb"><a href="#/analysis?id=${ev.backendId}">Analysis</a> ${ic('chevronRight')} <span>Tampering Map</span></div>
      <div class="page-sub" style="margin-top:-8px;">Evidence ID: <span class="mono">EV-${ev.backendId}</span> — ${escapeHtml(ev.filename)}</div>
      ${MayaVideoResult.renderVideoAttentionMap(a, {
        analysisId: a.analysisId,
        artifactUrl: MayaApi.analysis.artifactUrl,
        selectedFrame: requestedFrame,
      })}
    `;
    return appShell('investigator', '#/tampering', 'Tampering Map', inner);
  }
  const requestedFrame = params.get('frame');
  const selectedFrame = isVideoEvidence(ev)
    ? (requestedFrame !== null && requestedFrame !== '' && Number.isInteger(Number(requestedFrame))
      ? Number(requestedFrame)
      : defaultVideoFrameNumber(a))
    : null;
  window.__postRender = () => postRenderTampering(ev, a, selectedFrame);

  const confidence = typeof a.confidence === 'number' ? a.confidence : null;
  const xaiFrames = new Set(explainedFrameNumbers(a));
  const frameHasXai = selectedFrame === null
    ? !!(a.explanation && (a.explanation.heatmap || a.explanation.overlay))
    : xaiFrames.has(Number(selectedFrame));
  const hasHeatmap = frameHasXai && !!(a.explanation && a.explanation.heatmap);
  const hasOverlay = frameHasXai && !!(a.explanation && a.explanation.overlay);
  const originalUrl = videoOriginalUrl(ev, a, selectedFrame);
  const heatmapUrl = hasHeatmap
    ? MayaApi.analysis.artifactUrl(a.analysisId, 'heatmap', isVideoEvidence(ev) ? selectedFrame : undefined)
    : '';
  const overlayUrl = hasOverlay
    ? MayaApi.analysis.artifactUrl(a.analysisId, 'overlay', isVideoEvidence(ev) ? selectedFrame : undefined)
    : '';
  const selectedMeta = (a.frames || []).find((f) => Number(f.frameNumber) === Number(selectedFrame));

  /* All three views share one fixed-size stage; object-fit:contain keeps the
     real aspect ratio without stretching or cropping. */
  const imgTag = (src, alt, elId) => `<img id="${elId}" src="${src}" alt="${alt}" class="viewer-img"
      onerror="this.replaceWith(Object.assign(document.createElement('p'),{textContent:'Image could not be loaded from the backend.',className:'viewer-msg'}))">`;

  const inner = `
    <div class="breadcrumb"><a href="#/analysis?id=${ev.backendId}">Analysis</a> ${ic('chevronRight')} <span>Attention Heat Map</span></div>
    <div class="page-sub" style="margin-top:-8px;">Evidence ID: <span class="mono">EV-${ev.backendId}</span> — ${escapeHtml(ev.filename)}</div>

    ${isVideoEvidence(ev) ? videoFrameGalleryHtml(ev, a, selectedFrame, { links: false }) : ''}

    <div class="card">
      <div class="heatmap-tabs">
        <button class="heatmap-tab active" data-mode="original">ORIGINAL</button>
        <button class="heatmap-tab" data-mode="heatmap" ${hasHeatmap ? '' : 'disabled title="No heatmap artifact for this frame"'}>HEAT MAP</button>
        <button class="heatmap-tab" data-mode="overlay" ${hasOverlay ? '' : 'disabled title="No overlay artifact for this frame"'}>OVERLAY</button>
      </div>
      <div class="viewer-stage" id="heatStage">
        <div class="viewer-pane" data-pane="original">
          ${imgTag(originalUrl, isVideoEvidence(ev) ? 'Selected video frame' : 'Original evidence image', 'heatOriginal')}
          <span class="viewer-label" id="heatOriginalLabel">${isVideoEvidence(ev) && selectedFrame !== null ? `Frame ${selectedFrame}` : 'Original'}</span>
        </div>
        <div class="viewer-pane" data-pane="heatmap" hidden>
          ${hasHeatmap
            ? imgTag(heatmapUrl, 'Grad-CAM heat map', 'heatMap')
            : `<p class="viewer-msg">No heat map artifact is available for this frame.</p>`}
          <span class="viewer-label">Heat map</span>
        </div>
        <div class="viewer-pane" data-pane="overlay" hidden>
          ${hasOverlay
            ? imgTag(overlayUrl, 'Grad-CAM overlay', 'heatOverlay')
            : `<p class="viewer-msg">No overlay artifact is available for this frame.</p>`}
          <span class="viewer-label">Overlay</span>
        </div>
      </div>

      <div class="kv-grid" style="margin-top:20px;">
        <div class="kv-item"><div class="kl">${isVideoEvidence(ev) ? 'Video prediction' : 'Prediction'}</div><div class="kv-val">${escapeHtml(a.prediction || '—')}</div></div>
        <div class="kv-item"><div class="kl">${isVideoEvidence(ev) ? 'Video confidence' : 'Model confidence'}</div><div class="kv-val">${confidence !== null ? confidence.toFixed(1) + '%' : '—'}</div></div>
        <div class="kv-item"><div class="kl">Frame prediction</div><div class="kv-val" id="heatFramePred">${escapeHtml((selectedMeta && selectedMeta.prediction) || (isVideoEvidence(ev) ? '—' : (a.prediction || '—')))}</div></div>
        <div class="kv-item"><div class="kl">Frame confidence</div><div class="kv-val" id="heatFrameConf">${selectedMeta ? frameConfidenceLabel(selectedMeta) : (confidence !== null && !isVideoEvidence(ev) ? confidence.toFixed(1) + '%' : '—')}</div></div>
        <div class="kv-item"><div class="kl">Explainer</div><div class="kv-val">${escapeHtml((a.explanation && a.explanation.explainer) || 'none')}</div></div>
        <div class="kv-item"><div class="kl">Model</div><div class="kv-val">${escapeHtml(a.modelName || '—')}</div></div>
      </div>

      <div class="panel-title" style="margin-top:26px;"><h3>Flagged Regions</h3><span class="sub">Not available</span></div>
      <div class="empty-state">${ic('alert')}
        <p style="margin-top:10px;">
          MAYA's Grad-CAM output is a continuous pixel-attribution map, not a list of discrete regions.
          Region identifiers, bounding-box coordinates and per-region severity scores are not produced
          by the backend, so none are shown here. Read the heat map above for the areas that most
          influenced the verdict.
        </p>
      </div>
    </div>
  `;
  return appShell('investigator', '#/tampering', 'Attention Heat Map', inner);
}

function ensureViewerImage(pane, id, src, alt) {
  let img = document.getElementById(id);
  if (!img || img.tagName !== 'IMG') {
    pane.querySelectorAll('.viewer-msg').forEach((el) => el.remove());
    img = document.createElement('img');
    img.id = id;
    img.className = 'viewer-img';
    img.alt = alt;
    img.onerror = function () {
      this.replaceWith(Object.assign(document.createElement('p'), {
        textContent: 'Image could not be loaded from the backend.',
        className: 'viewer-msg',
      }));
    };
    pane.insertBefore(img, pane.firstChild);
  }
  img.src = src;
}

function showHeatMode(mode) {
  $$('.heatmap-tab').forEach((t) => t.classList.toggle('active', t.dataset.mode === mode));
  $$('#heatStage [data-pane]').forEach((pane) => {
    pane.hidden = pane.dataset.pane !== mode;
  });
}

function applyTamperingFrame(ev, a, frameNumber) {
  const xai = new Set(explainedFrameNumbers(a));
  const hasXai = xai.has(Number(frameNumber));
  const origPane = document.querySelector('#heatStage [data-pane="original"]');
  const heatPane = document.querySelector('#heatStage [data-pane="heatmap"]');
  const overPane = document.querySelector('#heatStage [data-pane="overlay"]');
  if (origPane) {
    ensureViewerImage(origPane, 'heatOriginal', MayaApi.evidence.frameUrl(ev.backendId, frameNumber), 'Selected video frame');
  }
  const label = document.getElementById('heatOriginalLabel');
  if (label) label.textContent = `Frame ${frameNumber}`;
  const heatTab = document.querySelector('.heatmap-tab[data-mode="heatmap"]');
  const overTab = document.querySelector('.heatmap-tab[data-mode="overlay"]');
  if (hasXai && a.explanation && a.explanation.heatmap) {
    heatTab.removeAttribute('disabled');
    heatTab.removeAttribute('title');
    if (heatPane) {
      ensureViewerImage(heatPane, 'heatMap', MayaApi.analysis.artifactUrl(a.analysisId, 'heatmap', frameNumber), 'Grad-CAM heat map');
    }
  } else if (heatTab) {
    heatTab.setAttribute('disabled', 'disabled');
    heatTab.title = 'No heatmap artifact for this frame';
  }
  if (hasXai && a.explanation && a.explanation.overlay) {
    overTab.removeAttribute('disabled');
    overTab.removeAttribute('title');
    if (overPane) {
      ensureViewerImage(overPane, 'heatOverlay', MayaApi.analysis.artifactUrl(a.analysisId, 'overlay', frameNumber), 'Grad-CAM overlay');
    }
  } else if (overTab) {
    overTab.setAttribute('disabled', 'disabled');
    overTab.title = 'No overlay artifact for this frame';
  }
  const frame = (a.frames || []).find((f) => Number(f.frameNumber) === Number(frameNumber));
  const predEl = document.getElementById('heatFramePred');
  const confEl = document.getElementById('heatFrameConf');
  if (predEl) predEl.textContent = (frame && frame.prediction) || '—';
  if (confEl) confEl.textContent = frame ? frameConfidenceLabel(frame) : '—';
  $$('.frame-tile').forEach((tile) => {
    tile.classList.toggle('selected', Number(tile.getAttribute('data-frame')) === Number(frameNumber));
  });
  const active = document.querySelector('.heatmap-tab.active');
  if (active && active.hasAttribute('disabled')) showHeatMode('original');
}

function postRenderTampering(ev, a) {
  $$('.heatmap-tab').forEach((tab) => tab.addEventListener('click', () => {
    if (tab.hasAttribute('disabled')) return;
    showHeatMode(tab.dataset.mode);
  }));
  if (!isVideoEvidence(ev)) return;
  $$('.frame-tile[data-frame]').forEach((tile) => {
    tile.addEventListener('click', () => {
      applyTamperingFrame(ev, a, Number(tile.getAttribute('data-frame')));
    });
  });
}

/* ==========================================================================
   15. CHAIN OF CUSTODY — password-protected lock + report
   ========================================================================== */
/* The old client-side custody password was security theatre: the check ran in
   the browser and the "secret" shipped in the bundle. Access is now decided by
   the backend, which authorises the caller against evidence ownership. */
async function renderCustodyLock() {
  const params = window.__routeParams;
  const id = params.get('id');
  if (!id) {
    return renderForensicCaseEvidencePicker({
      toolKey: 'custody',
      navPath: '#/chain-of-custody',
      title: 'Chain of Custody',
      subtitle: 'Select a case and evidence to view its MAYA audit / custody timeline.',
      actionLabel: 'Open Custody',
      buildHref: (e) => `#/custody-report?id=${e.backendId}&case=${e.caseBackendId || FORENSIC_SELECTION.custody.caseId || ''}`,
    });
  }
  location.hash = '#/custody-report?id=' + id + (params.get('case') ? `&case=${params.get('case')}` : '');
  return `<div class="page"></div>`;
}

/* Timeline is the real AuditLog trail for this evidence item. */
async function renderCustodyReport() {
  const params = window.__routeParams;
  const id = params.get('id');
  if (!id) {
    location.hash = '#/chain-of-custody';
    return `<div class="page"></div>`;
  }

  const custody = await MayaApi.evidence.custody(id);
  const ev = MayaApi.adapt.evidence(custody.evidence);
  const events = (custody.events || []).map(MayaApi.adapt.auditEvent);
  window.__custodyExport = { ev, events, custody };
  window.__postRender = () => postRenderCustodyReport(ev);

  const uploadEvent = events.find(e => e.eventType === 'EVIDENCE_UPLOADED');
  const verifyEvents = events.filter(e => e.eventType === 'EVIDENCE_VERIFIED');
  const lastVerify = verifyEvents[verifyEvents.length - 1];
  const integrityOk = lastVerify ? lastVerify.details.match !== false : null;

  const inner = `
    <div class="breadcrumb"><a href="#/analysis?id=${ev.backendId}">Analysis</a> ${ic('chevronRight')} <span>Chain of Custody</span></div>

    <div class="card" id="custodyReportPrintArea">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:14px;">
        <div class="brand"><span class="brand-mark">${ICONS.logo}</span><div>EVIDEX<small>CHAIN OF CUSTODY — MAYA AUDIT TRAIL</small></div></div>
        <span class="pill pill-green">${ic('lock')} Authorised by MAYA session</span>
      </div>
      <div class="divider"></div>
      <div class="kv-grid">
        <div class="kv-item"><div class="kl">Evidence ID</div><div class="kv-val">EV-${ev.backendId}</div></div>
        <div class="kv-item"><div class="kl">Case</div><div class="kv-val">${escapeHtml(custody.case_number || '—')}</div></div>
        <div class="kv-item"><div class="kl">Case title</div><div class="kv-val">${escapeHtml(custody.case_title || '—')}</div></div>
        <div class="kv-item"><div class="kl">Filename</div><div class="kv-val">${escapeHtml(ev.filename)}</div></div>
        <div class="kv-item"><div class="kl">MIME type</div><div class="kv-val">${escapeHtml(ev.mimeType || '—')}</div></div>
        <div class="kv-item"><div class="kl">Size</div><div class="kv-val">${fmtBytes(ev.size)}</div></div>
        <div class="kv-item"><div class="kl">Uploaded by</div><div class="kv-val">user#${ev.uploadedBy}</div></div>
        <div class="kv-item"><div class="kl">Uploaded at</div><div class="kv-val">${fmtDate(ev.timestamp)}</div></div>
        <div class="kv-item"><div class="kl">Evidence status</div><div class="kv-val">${escapeHtml(ev.evidenceStatus)}</div></div>
        <div class="kv-item"><div class="kl">Audit events</div><div class="kv-val">${custody.event_count}</div></div>
        <div class="kv-item"><div class="kl">Last integrity check</div><div class="kv-val">${
          lastVerify ? fmtDate(lastVerify.time) : 'Not re-verified since upload'}</div></div>
        <div class="kv-item"><div class="kl">Integrity result</div><div class="kv-val" style="color:var(--${integrityOk === null ? 'text-muted' : integrityOk ? 'green' : 'red'});">${
          integrityOk === null ? 'Hash recorded at upload' : integrityOk ? ic('checkCircle') + ' Hash matches' : ic('xCircle') + ' Hash mismatch'}</div></div>
      </div>

      <div class="panel-title" style="margin-top:22px;"><h3>SHA-256 (recorded at upload)</h3></div>
      <div class="hash-box"><span>${escapeHtml(ev.sha256)}</span></div>

      <div class="panel-title" style="margin-top:28px;"><h3>Custody Timeline</h3><span class="sub">MAYA audit log</span></div>
      <div class="timeline">
        ${events.length ? events.map(e => `
          <div class="tl-item">
            <div class="tl-dot"></div>
            <h4>${escapeHtml(e.message)}</h4>
            <div class="tl-meta">
              <span class="mono">${fmtDate(e.time)}</span>
              <span>${escapeHtml(e.actor)}</span>
              <span class="mono">${escapeHtml(e.eventType)}</span>
              <span class="pill pill-${e.severity === 'warning' ? 'amber' : 'green'}" style="padding:2px 8px;">${e.type}</span>
            </div>
          </div>`).join('')
        : `<p style="color:var(--text-muted);font-size:13.5px;">No audit events recorded for this evidence item.</p>`}
      </div>

      <p style="color:var(--text-muted);font-size:12px;margin-top:18px;">
        Physical custody fields (collection location, hand-over custodian, storage locker) are not tracked by MAYA and are therefore not shown.
      </p>
    </div>

    <div class="no-print" style="display:flex;gap:10px;margin-top:18px;flex-wrap:wrap;">
      <button class="btn btn-primary btn-sm" id="custodyDownloadBtn">${ic('download')} Download Timeline</button>
      <button class="btn btn-ghost btn-sm" id="custodyPrintBtn">${ic('print')} Print</button>
      <button class="btn btn-ghost btn-sm" data-action="verify-integrity" data-evidence="${ev.backendId}">${ic('security')} Re-verify Integrity</button>
      <a href="#/case-detail?id=${ev.caseBackendId}" class="btn btn-ghost btn-sm">Back to Case</a>
    </div>
  `;
  return appShell('investigator', '#/chain-of-custody', 'Chain of Custody', inner);
}

function postRenderCustodyReport(ev) {
  $('#custodyPrintBtn').addEventListener('click', () => window.print());
  $('#custodyDownloadBtn').addEventListener('click', () => {
    const { events, custody } = window.__custodyExport;
    const txt = [
      'MAYA / EVIDEX — CHAIN OF CUSTODY TIMELINE',
      '=========================================',
      `Exported: ${fmtDate(new Date())}`,
      '',
      `Evidence ID: EV-${ev.backendId}`,
      `Case: ${custody.case_number || '—'} — ${custody.case_title || '—'}`,
      `Filename: ${ev.filename}`,
      `MIME type: ${ev.mimeType || '—'}`,
      `Size: ${ev.size} bytes`,
      `SHA-256 (recorded at upload): ${ev.sha256}`,
      `Uploaded by: user#${ev.uploadedBy} at ${ev.timestamp}`,
      `Evidence status: ${ev.evidenceStatus}`,
      '',
      `AUDIT TIMELINE (${events.length} event${events.length === 1 ? '' : 's'})`,
      '--------------',
      ...(events.length
        ? events.map(e => `${e.time} — ${e.eventType} — ${e.actor} — ${e.message}${
            Object.keys(e.details).length ? ' — ' + JSON.stringify(e.details) : ''}`)
        : ['No audit events recorded.']),
      '',
      'Every line above is a stored MAYA audit record. Physical custody details',
      'are not tracked by the system and are omitted rather than estimated.',
    ].join('\n');
    downloadTextFile(`ChainOfCustody_EV-${ev.backendId}.txt`, txt);
    toast('Download started', `ChainOfCustody_EV-${ev.backendId}.txt`, 'success');
  });
}

/* ==========================================================================
   16. CASES  (list + detail)
   ========================================================================== */
/* Cases come from GET /api/cases; evidence counts are fetched per case so the
   list reflects real records rather than a cached array. */
async function renderCases() {
  const raw = await MayaApi.cases.list();
  const cases = raw.map(MayaApi.adapt.case);
  for (const c of cases) {
    try {
      const items = await MayaApi.evidence.listByCase(c.backendId);
      c.evidenceIds = items.map(e => String(e.evidence_id));
    } catch (err) {
      c.evidenceIds = [];
    }
  }
  window.__casesData = cases;
  window.__postRender = postRenderCases;

  const inner = `
    <p class="page-sub">Open cases in your forensic workspace.</p>
    <div class="card">
      <div class="table-toolbar">
        <div class="search-box">${ic('search')}<input type="text" id="caseSearch" placeholder="Search case number or title…"></div>
        <div class="filter-chips" id="caseFilters">
          <button class="filter-chip active" data-status="Open">Open</button>
          <button class="filter-chip" data-status="Under Review">Under Review</button>
        </div>
      </div>
      <div class="table-wrap">
        <table class="data-table">
          <thead><tr><th>Case Number</th><th>Title</th><th>Priority</th><th>Evidence</th><th>Status</th><th>Opened</th><th>Last Updated</th><th></th></tr></thead>
          <tbody id="casesTbody"></tbody>
        </table>
      </div>
      <p style="color:var(--text-muted);font-size:12px;margin:12px 4px 0;">
        Deleting a case removes its evidence records, analyses, face verifications and reports via
        <code>DELETE /api/cases/&lt;id&gt;</code>. The audit trail is retained.
      </p>
    </div>
  `;
  return appShell('investigator', '#/cases', 'Cases', inner);
}

function renderCasesRows(list) {
  $('#casesTbody').innerHTML = list.length ? list.map(c => `
    <tr>
      <td class="id-cell">${escapeHtml(c.caseNumber)}</td>
      <td>${escapeHtml(c.name)}</td>
      <td><span class="pill pill-${c.priority==='HIGH'||c.priority==='CRITICAL'?'red':c.priority==='MEDIUM'?'amber':'blue'}">${escapeHtml(c.priority || '—')}</span></td>
      <td>${c.evidenceIds.length}</td>
      <td><span class="pill pill-${c.status==='Open'?'blue':c.status==='Closed'?'muted':'amber'}">${escapeHtml(c.status)}</span></td>
      <td>${fmtDateShort(c.opened)}</td>
      <td>${fmtDateShort(c.updated)}</td>
      <td class="row-actions">
        <a href="#/case-detail?id=${c.backendId}" class="btn btn-ghost btn-sm">Open</a>
        <button class="btn btn-icon" data-delete-case="${c.backendId}"
                data-case-number="${escapeHtml(c.caseNumber)}" data-evidence-count="${c.evidenceIds.length}"
                title="Delete case ${escapeHtml(c.caseNumber)}">${ic('trash')}</button>
      </td>
    </tr>`).join('') : `<tr><td colspan="8"><div class="empty-state">No cases match this filter.</div></td></tr>`;
}

/* Real deletion against DELETE /api/cases/<id>. The backend enforces
   owner-or-admin authorization; on success the list is reloaded from the API
   rather than mutated locally. */
function bindCaseDeleteButtons(reload) {
  $$('[data-delete-case]').forEach(btn => btn.addEventListener('click', () => {
    const id = btn.dataset.deleteCase;
    const number = btn.dataset.caseNumber || `case ${id}`;
    const evCount = Number(btn.dataset.evidenceCount || 0);
    confirmModal(
      `Delete ${number}?`,
      `This permanently deletes the case and its ${evCount} evidence record(s), together with their analyses, ` +
      `face verifications and generated reports. The audit trail is kept. This cannot be undone.`,
      async () => {
        try {
          const res = await MayaApi.cases.remove(id);
          toast(
            'Case deleted',
            `${number} removed — ${res.deleted_evidence} evidence, ${res.deleted_analyses} analyses, ${res.deleted_reports} reports.`,
            'success', 6000,
          );
          await reload();
        } catch (err) {
          handleApiError(err, 'Case could not be deleted');
        }
      },
      'Delete case',
      true,
    );
  }));
}

function postRenderCases() {
  let status = 'Open', q = '';
  function apply() {
    let list = window.__casesData.filter(c => c.status !== 'Closed' && c.status !== 'Archived');
    if (status) list = list.filter(c => c.status === status);
    if (q) list = list.filter(c =>
      (c.caseNumber || '').toLowerCase().includes(q) || (c.name || '').toLowerCase().includes(q));
    renderCasesRows(list);
    bindCaseDeleteButtons(async () => { await router(); });
  }
  apply();
  $('#caseSearch').addEventListener('input', (e) => { q = e.target.value.toLowerCase(); apply(); });
  $$('#caseFilters .filter-chip').forEach(chip => chip.addEventListener('click', () => {
    $$('#caseFilters .filter-chip').forEach(c => c.classList.remove('active'));
    chip.classList.add('active'); status = chip.dataset.status; apply();
  }));
}

async function renderCaseDetail() {
  const params = window.__routeParams;
  const id = params.get('id');
  if (!id) { location.hash = '#/cases'; return `<div class="page"></div>`; }

  let c;
  try {
    c = MayaApi.adapt.case(await MayaApi.cases.get(id));
  } catch (err) {
    if (err instanceof MayaApi.ApiError && (err.isNotFound || err.isForbidden)) {
      return appShell('investigator', '#/cases', 'Case Detail',
        `<div class="empty-state card">${escapeHtml(err.message)} <a href="#/cases" style="color:var(--cyan);">Back to cases</a></div>`);
    }
    throw err;
  }

  // Attach each item's latest analysis so real verdicts can be listed.
  const items = await MayaApi.evidence.listByCase(c.backendId);
  const evidence = [];
  for (const raw of items) {
    const ev = MayaApi.adapt.evidence(raw, c.id);
    try {
      const runs = await MayaApi.evidence.analyses(ev.backendId);
      MayaApi.adapt.applyAnalysis(ev, runs.find(r => r.analysis_status === 'COMPLETED') || runs[0] || null);
    } catch (err) { /* leave as "Not analysed" */ }
    evidence.push(ev);
  }
  c.evidenceIds = evidence.map(e => e.id);
  window.__postRender = () => postRenderCaseDetail(c, evidence);

  const inner = `
    <div class="breadcrumb"><a href="#/cases">Cases</a> ${ic('chevronRight')} <span>${escapeHtml(c.caseNumber)}</span></div>
    <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;margin-bottom:18px;">
      <div>
        <h2 style="font-size:20px;">${escapeHtml(c.name)}</h2>
        <p style="color:var(--text-muted);font-size:13px;margin-top:4px;">${escapeHtml(c.caseNumber)} · Opened ${fmtDateShort(c.opened)}</p>
      </div>
      <div class="tag-row">
        <span class="pill pill-${c.priority==='HIGH'||c.priority==='CRITICAL'?'red':c.priority==='MEDIUM'?'amber':'blue'}">${escapeHtml(c.priority || '—')}</span>
        <span class="pill pill-${c.status==='Open'?'blue':c.status==='Closed'?'muted':'amber'}">${escapeHtml(c.status)}</span>
      </div>
    </div>

    <div class="tab-strip" id="caseTabs">
      ${['Overview','Evidence','Analysis','XAI','Heat Map','Chain of Custody','Reports'].map((t,i) => `<button class="tab-btn ${i===0?'active':''}" data-tab="${t}">${t}</button>`).join('')}
    </div>
    <div id="caseTabContent"></div>
  `;
  return appShell('investigator', '#/cases', 'Case Detail', inner);
}

function caseTabHtml(tab, c, evidence) {
  const analysed = evidence.filter(e => e.prediction);
  const link = (route, e, label) => `<div class="overview-card" style="border-bottom:1px solid var(--border-soft);border-radius:0;">
      <div class="oc-head">${ic(route === 'xai' ? 'xai' : route === 'tampering' ? 'tampering' : 'custody')}<h4>EV-${e.backendId} — ${escapeHtml(e.filename)}</h4></div>
      <a href="#/${route}?id=${e.backendId}" class="btn btn-ghost btn-sm">${label} ${ic('chevronRight')}</a></div>`;

  if (tab === 'Overview') return `
    <div class="stat-grid">
      <div class="stat-card"><div class="stat-label">Evidence Items</div><div class="stat-value">${evidence.length}</div></div>
      <div class="stat-card tone-red"><div class="stat-label">Tampered (FAKE)</div><div class="stat-value">${evidence.filter(e=>e.prediction==='FAKE').length}</div></div>
      <div class="stat-card tone-green"><div class="stat-label">Authentic (REAL)</div><div class="stat-value">${evidence.filter(e=>e.prediction==='REAL').length}</div></div>
      <div class="stat-card tone-amber"><div class="stat-label">Not Analysed</div><div class="stat-value">${evidence.length - analysed.length}</div></div>
    </div>
    <div class="card"><div class="panel-title"><h3>Case Summary</h3></div>
      <p style="color:var(--text-muted);font-size:13.5px;line-height:1.7;">
        ${escapeHtml(c.caseNumber)} was opened on ${fmtDateShort(c.opened)} and is currently <b style="color:var(--text);">${escapeHtml(c.status)}</b>
        at <b style="color:var(--text);">${escapeHtml(c.priority || 'unset')}</b> priority. It holds ${evidence.length} evidence item(s),
        ${analysed.length} of which have a completed MAYA authenticity analysis.
      </p>
      ${c.description ? `<p style="color:var(--text-muted);font-size:13.5px;line-height:1.7;">${escapeHtml(c.description)}</p>` : ''}
    </div>`;

  if (tab === 'Evidence') return `
    <div class="card"><div class="table-wrap"><table class="data-table">
      <thead><tr><th>Evidence ID</th><th>Filename</th><th>Type</th><th>Verdict</th><th>Confidence</th><th>Uploaded</th><th></th></tr></thead>
      <tbody>${evidence.length ? evidence.map(e => `<tr><td class="id-cell">EV-${e.backendId}</td><td>${escapeHtml(e.filename)}</td><td>${e.type}</td>
        <td>${statusPill(e.prediction ? e.status : e.analysisStatus)}</td>
        <td>${typeof e.confidence === 'number' ? e.confidence.toFixed(1) + '%' : '—'}</td>
        <td>${fmtDateShort(e.timestamp)}</td>
        <td><a href="#/analysis?id=${e.backendId}" class="btn btn-ghost btn-sm">View</a></td></tr>`).join('')
        : `<tr><td colspan="7" style="color:var(--text-muted);">No evidence in this case yet.</td></tr>`}</tbody>
    </table></div></div>`;

  if (tab === 'Analysis') return `
    <div class="card"><div class="panel-title"><h3>Analysis Summary</h3></div>
      <div class="region-list">${evidence.length ? evidence.map(e => `<div class="region-row">
        <span class="rr-dot" style="background:var(--${e.prediction==='FAKE'?'red':e.prediction==='REAL'?'green':'blue'})"></span>
        <div><div style="font-weight:600;font-size:13.5px;">${escapeHtml(e.filename)}</div><div class="rr-coords">EV-${e.backendId}${e.investigationId ? ' · ' + escapeHtml(e.investigationId) : ''}</div></div>
        ${statusPill(e.prediction ? e.status : e.analysisStatus)}
        <a href="#/analysis?id=${e.backendId}" class="btn btn-ghost btn-sm">Open</a></div>`).join('')
        : `<p style="color:var(--text-muted);font-size:13.5px;">No evidence in this case yet.</p>`}</div>
    </div>`;

  if (tab === 'XAI') return `
    <div class="card"><div class="panel-title"><h3>XAI Insights per Evidence</h3></div>
      ${evidence.length ? evidence.map(e => link('xai', e, 'View XAI Details')).join('')
        : `<p style="color:var(--text-muted);font-size:13.5px;">No evidence in this case yet.</p>`}
    </div>`;

  if (tab === 'Heat Map') return `
    <div class="card"><div class="panel-title"><h3>Grad-CAM Heat Map per Evidence</h3></div>
      ${evidence.length ? evidence.map(e => link('tampering', e, 'View Heat Map')).join('')
        : `<p style="color:var(--text-muted);font-size:13.5px;">No evidence in this case yet.</p>`}
    </div>`;

  if (tab === 'Chain of Custody') return `
    <div class="card"><div class="panel-title"><h3>Chain of Custody per Evidence</h3></div>
      ${evidence.length ? evidence.map(e => link('chain-of-custody', e, 'Open Audit Trail')).join('')
        : `<p style="color:var(--text-muted);font-size:13.5px;">No evidence in this case yet.</p>`}
    </div>`;

  if (tab === 'Reports') return `
    <div class="card"><div class="panel-title"><h3>Reports</h3></div>
      <p style="color:var(--text-muted);font-size:13.5px;margin-bottom:14px;">Forensic PDFs are generated per completed analysis.</p>
      ${analysed.length ? analysed.map(e => `<div class="overview-card" style="border-bottom:1px solid var(--border-soft);border-radius:0;">
          <div class="oc-head">${ic('reports')}<h4>EV-${e.backendId} — ${escapeHtml(e.filename)}</h4></div>
          <button class="btn btn-primary btn-sm" data-action="generate-report" data-analysis="${e.analysisId}">${ic('reports')} Generate PDF</button>
        </div>`).join('')
        : `<p style="color:var(--text-muted);font-size:13.5px;">No completed analyses to report on yet.</p>`}
      <a href="#/reports" class="btn btn-ghost btn-sm" style="margin-top:14px;">${ic('reports')} All reports</a>
    </div>`;
  return '';
}

function postRenderCaseDetail(c, evidence) {
  function renderTab(tab) { $('#caseTabContent').innerHTML = caseTabHtml(tab, c, evidence); }
  renderTab('Overview');
  $$('#caseTabs .tab-btn').forEach(btn => btn.addEventListener('click', () => {
    $$('#caseTabs .tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active'); renderTab(btn.dataset.tab);
  }));
}

/* ==========================================================================
   17. REPORTS
   ========================================================================== */
/* MAYA produces one report type: a forensic PDF per completed analysis. This
   page lists every real report generated across the caller's cases. */
async function renderReports(mode) {
  const cases = (await MayaApi.cases.list()).map(MayaApi.adapt.case);
  const rows = [];
  const pending = [];

  for (const c of cases) {
    const items = await MayaApi.evidence.listByCase(c.backendId);
    for (const raw of items) {
      const ev = MayaApi.adapt.evidence(raw, c.id);
      let runs = [];
      try { runs = await MayaApi.evidence.analyses(ev.backendId); } catch (err) { continue; }
      for (const run of runs) {
        if (run.analysis_status !== 'COMPLETED') continue;
        let reports = [];
        try { reports = await MayaApi.analysis.listReports(run.analysis_id); } catch (err) { reports = []; }
        if (reports.length) {
          reports.map(MayaApi.adapt.report).forEach(r => {
            rows.push({ report: r, ev, caseRef: c, prediction: run.prediction });
          });
        } else {
          pending.push({ analysisId: run.analysis_id, ev, caseRef: c, prediction: run.prediction });
        }
      }
    }
  }
  rows.sort((a, b) => new Date(b.report.generatedAt) - new Date(a.report.generatedAt));
  window.__postRender = postRenderReports;

  const inner = `
    <p class="page-sub">Forensic PDF reports generated by the MAYA report service.</p>

    <div class="card">
      <div class="panel-title"><h3>Generated Reports</h3><span class="sub">${rows.length} total</span></div>
      <div class="table-wrap">
        <table class="data-table">
          <thead><tr><th>Report Number</th><th>Case</th><th>Evidence</th><th>Verdict</th><th>Format</th><th>Size</th><th>Generated</th><th></th></tr></thead>
          <tbody>${rows.length ? rows.map(({ report: r, ev, caseRef, prediction }) => `
            <tr>
              <td class="id-cell">${escapeHtml(r.reportNumber)}</td>
              <td class="id-cell">${escapeHtml(caseRef.caseNumber)}</td>
              <td>${escapeHtml(ev.filename)}</td>
              <td>${statusPill(MayaApi.adapt.verdictFromPrediction(prediction))}</td>
              <td>${escapeHtml((r.format || 'pdf').toUpperCase())}</td>
              <td>${fmtBytes(r.sizeBytes || 0)}</td>
              <td>${fmtDateShort(r.generatedAt)}</td>
              <td class="row-actions">
                <button class="btn btn-ghost btn-sm" data-action="download-report" data-report="${r.reportId}" data-number="${escapeHtml(r.reportNumber)}">${ic('download')} PDF</button>
              </td>
            </tr>`).join('')
            : `<tr><td colspan="8" style="color:var(--text-muted);">No reports generated yet.</td></tr>`}
          </tbody>
        </table>
      </div>
    </div>

    <div class="card" style="margin-top:18px;">
      <div class="panel-title"><h3>Analyses Without a Report</h3><span class="sub">${pending.length} available</span></div>
      ${pending.length ? `<div style="display:flex;flex-direction:column;gap:8px;">
        ${pending.map(p => `
          <div class="overview-card" style="flex-direction:row;align-items:center;justify-content:space-between;padding:12px 14px;">
            <div>
              <div style="font-weight:600;font-size:13px;">${escapeHtml(p.ev.filename)}</div>
              <div style="font-size:11.5px;color:var(--text-dim);">${escapeHtml(p.caseRef.caseNumber)} · EV-${p.ev.backendId} · ${escapeHtml(p.prediction || '—')}</div>
            </div>
            <button class="btn btn-primary btn-sm" data-action="generate-report" data-analysis="${p.analysisId}">${ic('reports')} Generate PDF</button>
          </div>`).join('')}
      </div>` : `<p style="color:var(--text-muted);font-size:13.5px;margin:0;">Every completed analysis already has a report.</p>`}
    </div>

    <p style="color:var(--text-muted);font-size:12.5px;margin-top:14px;">
      MAYA generates a single consolidated forensic PDF per analysis, covering the authenticity verdict, integrity record, explainability and any face verification. Separate XAI-only, tampering-only and admissibility report types are not produced by the backend.
    </p>
  `;
  return appShell(mode === 'admin' ? 'admin' : 'investigator', mode === 'admin' ? '#/admin-reports' : '#/reports', 'Reports', inner);
}

function postRenderReports() {
  $$('[data-report-print]').forEach(b => b.addEventListener('click', () => window.print()));
}

/* ==========================================================================
   18. LEGAL ADMISSIBILITY
   ========================================================================== */
/* Evidence readiness checklist — every row is a verifiable fact from the
   database. MAYA does not determine legal admissibility, and the page no
   longer claims to. */
async function renderLegalAdmissibility() {
  const params = window.__routeParams;
  const id = params.get('id');
  if (!id) {
    return appShell('investigator', '#/legal-admissibility', 'Evidence Readiness',
      `<div class="empty-state card">${ic('legal')}<p style="margin-top:10px;">Open an evidence item from <a href="#/cases" style="color:var(--cyan);">your cases</a> to see its readiness checklist.</p></div>`);
  }

  const custody = await MayaApi.evidence.custody(id);
  const ev = MayaApi.adapt.evidence(custody.evidence);
  const events = (custody.events || []).map(MayaApi.adapt.auditEvent);

  let runs = [];
  try { runs = await MayaApi.evidence.analyses(id); } catch (err) { runs = []; }
  const completed = runs.find(r => r.analysis_status === 'COMPLETED') || null;

  let reports = [];
  if (completed) {
    try { reports = await MayaApi.analysis.listReports(completed.analysis_id); } catch (err) { reports = []; }
  }

  const verifyEvents = events.filter(e => e.eventType === 'EVIDENCE_VERIFIED');
  const lastVerify = verifyEvents[verifyEvents.length - 1];
  const integrityVerified = lastVerify ? lastVerify.details.match !== false : null;
  const videoReadiness = completed && completed.video_analysis && window.MayaVideoResult
    ? MayaVideoResult.evidenceReadiness(completed)
    : null;

  const checks = [
    { label: 'Evidence stored in MAYA', ok: true, note: `Evidence EV-${ev.backendId}, uploaded ${fmtDate(ev.timestamp)}` },
    { label: 'SHA-256 recorded at ingest', ok: !!ev.sha256, note: ev.sha256 ? ev.sha256.slice(0, 32) + '…' : 'No hash on record' },
    { label: 'Integrity re-verified against stored hash', ok: integrityVerified === true,
      note: integrityVerified === null ? 'Never re-verified — run a verification to confirm the file is unchanged'
        : integrityVerified ? `Hash matched on ${fmtDate(lastVerify.time)}` : 'Hash MISMATCH — file differs from ingest' },
    { label: 'Uploader identified', ok: !!ev.uploadedBy, note: `user#${ev.uploadedBy}` },
    { label: 'Linked to a case', ok: !!custody.case_number, note: `${custody.case_number || '—'} — ${custody.case_title || '—'}` },
    { label: 'Audit trail present', ok: events.length > 0, note: `${events.length} recorded event(s)` },
    { label: 'Authenticity analysis completed', ok: !!completed,
      note: videoReadiness
        ? videoReadiness.analysisNote
        : (completed ? `${completed.prediction} at ${Number(completed.confidence).toFixed(1)}% confidence (${completed.model_name} ${completed.model_version})` : 'No completed analysis') },
    { label: 'Explainability artifact produced',
      ok: videoReadiness
        ? videoReadiness.explainabilityOk
        : !!(completed && completed.explanation && (completed.explanation.heatmap || completed.explanation.overlay)),
      note: videoReadiness
        ? videoReadiness.explainabilityNote
        : (completed && completed.explanation && completed.explanation.heatmap ? `Grad-CAM (${completed.explanation.explainer})` : 'No Grad-CAM artifact') },
    { label: 'Forensic report generated', ok: reports.length > 0,
      note: reports.length ? `${reports[0].report_number} (${fmtDate(reports[0].generated_at)})` : 'No report generated yet' },
  ];
  const passCount = checks.filter(c => c.ok).length;
  const ready = passCount === checks.length;

  const inner = `
    <div class="breadcrumb"><a href="#/analysis?id=${ev.backendId}">Analysis</a> ${ic('chevronRight')} <span>Evidence Readiness</span></div>
    <p class="page-sub">Evidence ID: <span class="mono">EV-${ev.backendId}</span> — ${escapeHtml(ev.filename)}</p>
    <div class="grid-2">
      <div class="card">
        <div class="panel-title"><h3>Evidence Readiness Checklist</h3><span class="sub">system facts only</span></div>
        <div class="checklist">
          ${checks.map(c => `
            <div class="checklist-item">
              <div class="check-mark ${c.ok?'':'pending'}">${ic(c.ok?'check':'alert')}</div>
              <div class="ci-text">${escapeHtml(c.label)}${c.note?`<div class="ci-note">${escapeHtml(c.note)}</div>`:''}</div>
            </div>`).join('')}
        </div>
      </div>
      <div>
        <div class="verdict-banner card" style="border-color:var(--${ready?'green':'amber'});">
          <div style="color:var(--text-muted);font-size:13px;">EVIDENCE RECORD COMPLETENESS</div>
          <div class="vb-status" style="color:var(--${ready?'green':'amber'});">${ready ? 'COMPLETE' : 'INCOMPLETE'}</div>
          <div style="color:var(--text-dim);font-size:12.5px;margin-top:6px;">${passCount}/${checks.length} records present</div>
        </div>
        <div class="disclaimer-box">${ic('gavel')}
          <strong>This is not a legal determination.</strong> MAYA does not and cannot decide whether evidence is
          admissible in court. The list above only reports which records exist in the system: a hash, an audit
          trail, a model prediction and a report. Admissibility is decided by a court, applying rules of evidence
          to the underlying material and its handling.
        </div>
        <div class="card" style="margin-top:14px;display:flex;gap:10px;flex-wrap:wrap;">
          <button class="btn btn-ghost btn-sm" data-action="verify-integrity" data-evidence="${ev.backendId}">${ic('security')} Re-verify Integrity</button>
          <a href="#/chain-of-custody?id=${ev.backendId}" class="btn btn-ghost btn-sm">${ic('custody')} Audit Trail</a>
          ${completed ? `<button class="btn btn-ghost btn-sm" data-action="generate-report" data-analysis="${completed.analysis_id}">${ic('reports')} Generate PDF</button>` : ''}
        </div>
      </div>
    </div>
  `;
  return appShell('investigator', '#/legal-admissibility', 'Evidence Readiness', inner);
}

/* ==========================================================================
   19. PROFILE
   ========================================================================== */
async function renderProfile() {
  const user = await MayaApi.auth.me();
  const stats = await MayaApi.dashboard.stats();
  const name = user.full_name || user.username;
  const inner = `
    <div class="card" style="max-width:560px;">
      <div style="display:flex;gap:16px;align-items:center;">
        <div class="user-avatar" style="width:60px;height:60px;font-size:20px;">${initials(name)}</div>
        <div><h2 style="font-size:19px;">${escapeHtml(name)}</h2><p style="color:var(--text-muted);font-size:13px;">${escapeHtml(user.role)} · @${escapeHtml(user.username)}</p></div>
      </div>
      <div class="divider"></div>
      <div class="kv-grid">
        <div class="kv-item"><div class="kl">Role</div><div class="kv-val">${escapeHtml(user.role)}</div></div>
        <div class="kv-item"><div class="kl">Email</div><div class="kv-val">${escapeHtml(user.email)}</div></div>
        <div class="kv-item"><div class="kl">Account status</div><div class="kv-val">${user.is_active ? 'Active' : 'Inactive'}</div></div>
        <div class="kv-item"><div class="kl">Registered</div><div class="kv-val">${user.created_at ? fmtDateShort(user.created_at) : '—'}</div></div>
        <div class="kv-item"><div class="kl">Last login</div><div class="kv-val">${user.last_login_at ? fmtDate(user.last_login_at) : '—'}</div></div>
        <div class="kv-item"><div class="kl">Session</div><div class="kv-val">MAYA server session</div></div>
        <div class="kv-item"><div class="kl">Cases</div><div class="kv-val">${stats.counts.cases}</div></div>
        <div class="kv-item"><div class="kl">Evidence</div><div class="kv-val">${stats.counts.evidence}</div></div>
      </div>
      <div class="divider"></div>
      <button class="btn btn-danger" data-action="logout">${ic('logout')} Logout</button>
    </div>
  `;
  return appShell('investigator', '#/profile', 'Profile', inner);
}

/* ==========================================================================
   20. ADMIN DASHBOARD
   ========================================================================== */
/* Every figure below comes from GET /api/dashboard/stats (admin scope = whole
   database), GET /api/admin/users and GET /api/audit. Nothing is seeded and
   nothing is read from localStorage. */
async function renderAdminDashboard() {
  const [stats, users, auditRaw] = await Promise.all([
    MayaApi.dashboard.stats(),
    MayaApi.admin.users(),
    MayaApi.audit.list(),
  ]);
  const events = auditRaw.map(MayaApi.adapt.auditEvent);
  const roleCount = (role) => users.filter(u => u.role === role).length;
  const activeInvestigators = users.filter(u => u.role === 'INVESTIGATOR' && u.is_active).length;
  const fakeCount = stats.prediction.FAKE || 0;

  window.__postRender = () => postRenderAdminDashboard(stats, users);

  const inner = `
    <p class="page-sub">System-wide oversight of users, cases and evidence — live backend data (scope: ${escapeHtml(stats.scope)}).</p>
    <div class="stat-grid">
      <div class="stat-card"><div class="stat-label">Total Users</div><div class="stat-value">${users.length}</div></div>
      <div class="stat-card tone-blue"><div class="stat-label">Active Investigators</div><div class="stat-value">${activeInvestigators}</div></div>
      <div class="stat-card"><div class="stat-label">Total Cases</div><div class="stat-value">${stats.counts.cases}</div></div>
      <div class="stat-card tone-green"><div class="stat-label">Evidence Stored</div><div class="stat-value">${stats.counts.evidence}</div></div>
      <div class="stat-card tone-red"><div class="stat-label">Classified FAKE</div><div class="stat-value">${fakeCount}</div></div>
      <div class="stat-card tone-amber"><div class="stat-label">Reports Generated</div><div class="stat-value">${stats.counts.reports}</div></div>
    </div>
    <div class="stat-grid" style="margin-top:0;">
      <div class="stat-card"><div class="stat-label">Analyses Run</div><div class="stat-value">${stats.counts.analyses}</div></div>
      <div class="stat-card tone-green"><div class="stat-label">Analyses Completed</div><div class="stat-value">${stats.counts.analyses_completed}</div></div>
      <div class="stat-card tone-blue"><div class="stat-label">Face Verifications</div><div class="stat-value">${stats.counts.face_verifications}</div></div>
      <div class="stat-card"><div class="stat-label">Audit Events (latest 200)</div><div class="stat-value">${events.length}</div></div>
    </div>

    <div class="grid-charts">
      <div class="card">
        <div class="panel-title"><h3>Authenticity Verdicts</h3><span class="sub">Completed analyses, last 14 days</span></div>
        <div class="chart-wrap"><canvas id="adminChartActivity"></canvas></div>
      </div>
      <div class="card">
        <div class="panel-title"><h3>User Roles</h3><span class="sub">Real accounts</span></div>
        <div class="chart-wrap"><canvas id="adminChartRoles"></canvas></div>
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="panel-title"><h3>Accounts</h3><a href="#/admin-users" class="btn btn-ghost btn-sm">View all</a></div>
        <div class="table-wrap"><table class="data-table">
          <thead><tr><th>Name</th><th>Username</th><th>Role</th><th>Status</th></tr></thead>
          <tbody>${users.slice(0, 5).map(u => `<tr>
            <td>${escapeHtml(u.full_name || '—')}</td>
            <td class="mono">${escapeHtml(u.username)}</td>
            <td>${escapeHtml(u.role)}</td>
            <td><span class="pill pill-${u.is_active ? 'green' : 'red'}">${u.is_active ? 'Active' : 'Inactive'}</span></td>
          </tr>`).join('')}</tbody>
        </table></div>
      </div>
      <div class="card">
        <div class="panel-title"><h3>Recent Audit Events</h3><a href="#/admin-logs" class="btn btn-ghost btn-sm">View all</a></div>
        <div style="display:flex;flex-direction:column;gap:10px;">
          ${events.length ? events.slice(0, 6).map(l => `
            <div style="display:flex;justify-content:space-between;gap:10px;font-size:12.5px;padding:10px 0;border-bottom:1px solid var(--border-soft);">
              <span><span class="pill pill-blue" style="margin-right:8px;">${escapeHtml(l.type)}</span>${escapeHtml(l.message)}</span>
              <span class="mono" style="color:var(--text-dim);">${fmtDateShort(l.time)}</span>
            </div>`).join('') : `<p style="color:var(--text-muted);font-size:13px;margin:0;">No audit events recorded yet.</p>`}
        </div>
      </div>
    </div>

    <div class="card">
      <div class="panel-title"><h3>Latest Analyses</h3><span class="sub">From stored analysis runs</span></div>
      <div class="table-wrap"><table class="data-table">
        <thead><tr><th>Analysis</th><th>Evidence</th><th>Case</th><th>Investigation ID</th><th>Prediction</th><th>Confidence</th><th>Status</th><th>Completed</th></tr></thead>
        <tbody>${stats.recent_analyses.length ? stats.recent_analyses.map(r => `<tr>
          <td class="id-cell">${r.analysis_id}</td>
          <td class="mono">EV-${r.evidence_id}</td>
          <td class="mono">${r.case_id}</td>
          <td class="mono">${escapeHtml(r.investigation_id || '—')}</td>
          <td>${escapeHtml(r.prediction || '—')}</td>
          <td>${r.confidence === null || r.confidence === undefined ? '—' : (Number(r.confidence) * 100).toFixed(1) + '%'}</td>
          <td>${escapeHtml(r.status || '—')}</td>
          <td>${r.completed_at ? fmtDate(r.completed_at) : '—'}</td>
        </tr>`).join('') : `<tr><td colspan="8"><div class="empty-state">No analyses have been run yet.</div></td></tr>`}</tbody>
      </table></div>
    </div>
  `;
  return appShell('admin', '#/admin', 'Admin Dashboard', inner);
}

function postRenderAdminDashboard(stats, users) {
  const labels = stats.trend.map(d => new Date(d.date + 'T00:00:00').toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
  new Chart($('#adminChartActivity'), {
    type: 'bar',
    data: { labels, datasets: [
      { label: 'REAL', data: stats.trend.map(d => d.real), backgroundColor: '#2ecc9b', borderRadius: 4, maxBarThickness: 20 },
      { label: 'FAKE', data: stats.trend.map(d => d.fake), backgroundColor: '#f1495b', borderRadius: 4, maxBarThickness: 20 },
    ]},
    options: { responsive:true, maintainAspectRatio:false, plugins:{legend:{position:'bottom',labels:{color:'#8ca0b3',boxWidth:10,font:{size:11}}}}, scales:{x:{stacked:true,grid:{display:false},ticks:{color:'#5b6c7d',font:{size:10}}},y:{stacked:true,grid:{color:'#1e2a38'},ticks:{color:'#5b6c7d',precision:0},beginAtZero:true}} }
  });

  const roles = {};
  users.forEach(u => { roles[u.role] = (roles[u.role] || 0) + 1; });
  new Chart($('#adminChartRoles'), {
    type: 'doughnut',
    data: { labels: Object.keys(roles), datasets: [{ data: Object.values(roles), backgroundColor:['#f5a623','#2e8fe0','#29e0d6','#8b5cf6'], borderWidth:0 }]},
    options: { responsive:true, maintainAspectRatio:false, cutout:'68%', plugins:{legend:{position:'bottom',labels:{color:'#8ca0b3',boxWidth:10,font:{size:11}}}} }
  });
}

/* ==========================================================================
   21. ADMIN USERS / INVESTIGATORS
   ========================================================================== */
/* Real accounts from GET /api/admin/users (ADMIN-only on the backend). The
   backend exposes no create/suspend user endpoint, so this page is read-only
   rather than offering demo write actions that would not persist. */
async function renderAdminUsers(roleFilter) {
  const users = await MayaApi.admin.users();
  const list = roleFilter === 'Investigator' ? users.filter(u => u.role === 'INVESTIGATOR') : users;
  const title = roleFilter === 'Investigator' ? 'Investigators' : 'Users';
  window.__adminUsers = list;
  window.__postRender = postRenderAdminUsers;

  const inner = `
    <p class="page-sub">${list.length} account(s) in the backend user table.</p>
    <div class="card">
      <div class="table-toolbar">
        <div class="search-box">${ic('search')}<input type="text" id="userSearch" placeholder="Search by name or username…"></div>
      </div>
      <div class="table-wrap"><table class="data-table">
        <thead><tr><th>ID</th><th>Name</th><th>Username</th><th>Email</th><th>Role</th><th>Status</th><th>Cases</th><th>Evidence</th><th>Registered</th><th>Last Login</th></tr></thead>
        <tbody id="usersTbody"></tbody>
      </table></div>
      <p style="color:var(--text-muted);font-size:12px;margin:12px 4px 0;">
        Read-only: the backend has no endpoint for creating or suspending accounts. New accounts are created through
        <code>POST /api/auth/register</code>.
      </p>
    </div>
  `;
  return appShell('admin', roleFilter === 'Investigator' ? '#/admin-investigators' : '#/admin-users', title, inner);
}

function renderUsersRows(list) {
  $('#usersTbody').innerHTML = list.length ? list.map(u => `
    <tr>
      <td class="id-cell">${u.id}</td>
      <td>${escapeHtml(u.full_name || '—')}</td>
      <td class="mono">${escapeHtml(u.username)}</td>
      <td class="mono">${escapeHtml(u.email)}</td>
      <td>${escapeHtml(u.role)}</td>
      <td><span class="pill pill-${u.is_active ? 'green' : 'red'}">${u.is_active ? 'Active' : 'Inactive'}</span></td>
      <td>${u.case_count}</td>
      <td>${u.evidence_count}</td>
      <td>${u.created_at ? fmtDateShort(u.created_at) : '—'}</td>
      <td>${u.last_login_at ? fmtDate(u.last_login_at) : 'Never'}</td>
    </tr>`).join('') : `<tr><td colspan="10"><div class="empty-state">No accounts found.</div></td></tr>`;
}

function postRenderAdminUsers() {
  const all = window.__adminUsers || [];
  renderUsersRows(all);
  $('#userSearch').addEventListener('input', (e) => {
    const q = e.target.value.toLowerCase();
    renderUsersRows(all.filter(u =>
      (u.full_name || '').toLowerCase().includes(q) ||
      u.username.toLowerCase().includes(q) ||
      u.email.toLowerCase().includes(q)));
  });
}

/* ==========================================================================
   22. ADMIN CASES / EVIDENCE / ACTIVITY / LOGS / SECURITY / SETTINGS
   ========================================================================== */
/* Every case in the database (GET /api/cases returns all cases for ADMIN),
   with real evidence counts and a working delete action. */
async function renderAdminCases() {
  const raw = await MayaApi.cases.list();
  const cases = raw.map(MayaApi.adapt.case);
  for (const c of cases) {
    try {
      const items = await MayaApi.evidence.listByCase(c.backendId);
      c.evidenceIds = items.map(e => String(e.evidence_id));
    } catch (err) { c.evidenceIds = []; }
  }
  window.__casesData = cases;
  window.__postRender = postRenderAdminCases;

  const inner = `
    <p class="page-sub">${cases.length} case(s) across all investigators.</p>
    <div class="card">
      <div class="table-toolbar">
        <div class="search-box">${ic('search')}<input type="text" id="caseSearch" placeholder="Search case number or title…"></div>
        <div class="filter-chips" id="caseFilters">
          <button class="filter-chip active" data-status="All">All</button>
          <button class="filter-chip" data-status="Open">Open</button>
          <button class="filter-chip" data-status="Under Review">Under Review</button>
        </div>
      </div>
      <div class="table-wrap"><table class="data-table">
        <thead><tr><th>Case Number</th><th>Title</th><th>Owner</th><th>Priority</th><th>Evidence</th><th>Status</th><th>Last Updated</th><th></th></tr></thead>
        <tbody id="adminCasesTbody"></tbody>
      </table></div>
    </div>
  `;
  return appShell('admin', '#/admin-cases', 'Cases', inner);
}

function postRenderAdminCases() {
  let status = 'All', q = '';
  function apply() {
    let list = window.__casesData.filter(c => c.status !== 'Closed' && c.status !== 'Archived');
    if (status !== 'All') list = list.filter(c => c.status === status);
    if (q) list = list.filter(c =>
      (c.caseNumber || '').toLowerCase().includes(q) || (c.name || '').toLowerCase().includes(q));
    $('#adminCasesTbody').innerHTML = list.length ? list.map(c => `
      <tr>
        <td class="id-cell">${escapeHtml(c.caseNumber)}</td>
        <td>${escapeHtml(c.name)}</td>
        <td class="mono">user#${escapeHtml(String(c.createdBy ?? '—'))}</td>
        <td><span class="pill pill-${c.priority==='HIGH'||c.priority==='CRITICAL'?'red':c.priority==='MEDIUM'?'amber':'blue'}">${escapeHtml(c.priority || '—')}</span></td>
        <td>${c.evidenceIds.length}</td>
        <td><span class="pill pill-${c.status==='Open'?'blue':'amber'}">${escapeHtml(c.status)}</span></td>
        <td>${fmtDateShort(c.updated)}</td>
        <td class="row-actions">
          <a href="#/case-detail?id=${c.backendId}" class="btn btn-ghost btn-sm">Open</a>
          <button class="btn btn-icon" data-delete-case="${c.backendId}"
                  data-case-number="${escapeHtml(c.caseNumber)}" data-evidence-count="${c.evidenceIds.length}"
                  title="Delete case ${escapeHtml(c.caseNumber)}">${ic('trash')}</button>
        </td>
      </tr>`).join('') : `<tr><td colspan="8"><div class="empty-state">No cases match this filter.</div></td></tr>`;
    bindCaseDeleteButtons(async () => { await router(); });
  }
  apply();
  $('#caseSearch').addEventListener('input', (e) => { q = e.target.value.toLowerCase(); apply(); });
  $$('#caseFilters .filter-chip').forEach(chip => chip.addEventListener('click', () => {
    $$('#caseFilters .filter-chip').forEach(c => c.classList.remove('active'));
    chip.classList.add('active'); status = chip.dataset.status; apply();
  }));
}

/* Real evidence, collected per case from GET /api/evidence/cases/<id> and
   annotated with each item's latest stored analysis. */
async function renderAdminEvidence() {
  const cases = (await MayaApi.cases.list()).map(MayaApi.adapt.case);
  const rows = [];
  for (const c of cases) {
    let items = [];
    try { items = await MayaApi.evidence.listByCase(c.backendId); } catch (err) { continue; }
    for (const rawEv of items) {
      const ev = MayaApi.adapt.evidence(rawEv, c.id);
      try {
        const runs = await MayaApi.evidence.analyses(ev.backendId);
        MayaApi.adapt.applyAnalysis(ev, runs.find(r => r.analysis_status === 'COMPLETED') || runs[0] || null);
      } catch (err) { /* leave unanalysed */ }
      ev.caseNumber = c.caseNumber;
      rows.push(ev);
    }
  }
  window.__adminEvidence = rows;
  window.__postRender = postRenderAdminEvidence;

  const inner = `
    <p class="page-sub">${rows.length} evidence item(s) stored across all cases.</p>
    <div class="card">
      <div class="table-toolbar">
        <div class="search-box">${ic('search')}<input type="text" id="evSearch" placeholder="Search filename or evidence ID…"></div>
      </div>
      <div class="table-wrap"><table class="data-table">
        <thead><tr><th>Evidence ID</th><th>Filename</th><th>Case</th><th>Type</th><th>Size</th><th>Status</th><th>Prediction</th><th>Uploaded</th></tr></thead>
        <tbody id="evTbody"></tbody>
      </table></div>
    </div>
  `;
  return appShell('admin', '#/admin-evidence', 'Evidence', inner);
}

function postRenderAdminEvidence() {
  const all = window.__adminEvidence || [];
  function rows(list) {
    $('#evTbody').innerHTML = list.length ? list.map(e => `<tr>
      <td class="id-cell">EV-${e.backendId}</td>
      <td>${escapeHtml(e.filename)}</td>
      <td class="mono">${escapeHtml(e.caseNumber || '—')}</td>
      <td>${escapeHtml(e.type || '—')}</td>
      <td>${e.size ? fmtBytes(e.size) : '—'}</td>
      <td>${statusPill(e.status)}</td>
      <td>${escapeHtml(e.prediction || 'Not analysed')}</td>
      <td>${fmtDateShort(e.timestamp)}</td>
    </tr>`).join('') : `<tr><td colspan="8"><div class="empty-state">No evidence has been uploaded yet.</div></td></tr>`;
  }
  rows(all);
  $('#evSearch').addEventListener('input', (e) => {
    const q = e.target.value.toLowerCase();
    rows(all.filter(x => (x.filename || '').toLowerCase().includes(q) || String(x.backendId).includes(q)));
  });
}

/* Real 14-day authenticity trend and the stored recent analysis runs. */
async function renderAdminActivity() {
  const stats = await MayaApi.dashboard.stats();
  window.__postRender = () => postRenderAdminActivity(stats);
  const inner = `
    <p class="page-sub">Completed analyses per day over the last 14 days, from stored analysis timestamps.</p>
    <div class="card"><div class="chart-wrap" style="height:280px;"><canvas id="adminActivityChart"></canvas></div></div>
    <div class="card" style="margin-top:18px;">
      <div class="panel-title"><h3>Latest Analysis Runs</h3><span class="sub">${stats.counts.analyses} total</span></div>
      <div style="display:flex;flex-direction:column;gap:2px;">
        ${stats.recent_analyses.length ? stats.recent_analyses.map(r => `
          <div style="display:flex;justify-content:space-between;gap:10px;padding:12px 0;border-bottom:1px solid var(--border-soft);font-size:13px;">
            <span><span class="mono" style="color:var(--text-dim);">EV-${r.evidence_id}</span> — ${escapeHtml(r.investigation_id || 'no investigation id')}</span>
            <span>${escapeHtml(r.prediction || r.status || '—')}${r.confidence !== null && r.confidence !== undefined ? ` · ${(Number(r.confidence) * 100).toFixed(1)}%` : ''}</span>
          </div>`).join('') : `<p style="color:var(--text-muted);font-size:13px;margin:0;">No analyses have been run yet.</p>`}
      </div>
    </div>
  `;
  return appShell('admin', '#/admin-activity', 'Analysis Activity', inner);
}

function postRenderAdminActivity(stats) {
  const labels = stats.trend.map(d => new Date(d.date + 'T00:00:00').toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
  new Chart($('#adminActivityChart'), {
    type: 'line',
    data: { labels, datasets: [
      { label:'REAL', data: stats.trend.map(d => d.real), borderColor:'#2ecc9b', backgroundColor:'rgba(46,204,155,0.10)', fill:true, tension:0.35, pointRadius:2 },
      { label:'FAKE', data: stats.trend.map(d => d.fake), borderColor:'#f1495b', backgroundColor:'rgba(241,73,91,0.10)', fill:true, tension:0.35, pointRadius:2 },
    ]},
    options: { responsive:true, maintainAspectRatio:false, plugins:{legend:{position:'bottom',labels:{color:'#8ca0b3',boxWidth:10,font:{size:11}}}}, scales:{x:{grid:{color:'#1e2a38'},ticks:{color:'#5b6c7d',font:{size:10}}},y:{grid:{color:'#1e2a38'},ticks:{color:'#5b6c7d',precision:0},beginAtZero:true}} }
  });
}

/* The real server-side audit trail (GET /api/audit, 200 most recent). */
async function renderAdminLogs() {
  const events = (await MayaApi.audit.list()).map(MayaApi.adapt.auditEvent);
  window.__adminLogs = events;
  window.__postRender = postRenderAdminLogs;
  const inner = `
    <p class="page-sub">Server-side audit log — ${events.length} most recent event(s).</p>
    <div class="card">
      <div class="table-toolbar">
        <div class="search-box">${ic('search')}<input type="text" id="logSearch" placeholder="Search event type, message or actor…"></div>
      </div>
      <div class="table-wrap"><table class="data-table">
        <thead><tr><th>Audit ID</th><th>Category</th><th>Event</th><th>Actor</th><th>Case</th><th>Evidence</th><th>Time</th></tr></thead>
        <tbody id="logsTbody"></tbody>
      </table></div>
    </div>
  `;
  return appShell('admin', '#/admin-logs', 'System Logs', inner);
}

function postRenderAdminLogs() {
  const all = window.__adminLogs || [];
  function rows(list) {
    $('#logsTbody').innerHTML = list.length ? list.map(l => `<tr>
      <td class="id-cell">${escapeHtml(l.id)}</td>
      <td><span class="pill pill-blue">${escapeHtml(l.type)}</span></td>
      <td class="mono">${escapeHtml(l.eventType)}</td>
      <td class="mono">${escapeHtml(l.actor)}</td>
      <td class="mono">${l.caseId ?? '—'}</td>
      <td class="mono">${l.evidenceId ? 'EV-' + l.evidenceId : '—'}</td>
      <td>${fmtDate(l.time)}</td>
    </tr>`).join('') : `<tr><td colspan="7"><div class="empty-state">No audit events recorded yet.</div></td></tr>`;
  }
  rows(all);
  $('#logSearch').addEventListener('input', (e) => {
    const q = e.target.value.toLowerCase();
    rows(all.filter(l =>
      (l.eventType || '').toLowerCase().includes(q) ||
      (l.message || '').toLowerCase().includes(q) ||
      (l.actor || '').toLowerCase().includes(q)));
  });
}

/* Authentication and access events taken from the real audit trail. */
async function renderAdminSecurity() {
  const [users, auditRaw] = await Promise.all([MayaApi.admin.users(), MayaApi.audit.list()]);
  const events = auditRaw.map(MayaApi.adapt.auditEvent);
  const AUTH_EVENTS = ['USER_LOGIN', 'USER_LOGOUT', 'USER_REGISTERED'];
  const ACCESS_EVENTS = ['EVIDENCE_ACCESSED', 'EVIDENCE_VERIFIED', 'CASE_DELETED'];
  const authEvents = events.filter(e => AUTH_EVENTS.includes(e.eventType));
  const sensitive = events.filter(e => AUTH_EVENTS.includes(e.eventType) || ACCESS_EVENTS.includes(e.eventType));

  const inner = `
    <p class="page-sub">Authentication and access events recorded by the backend audit log.</p>
    <div class="stat-grid">
      <div class="stat-card tone-blue"><div class="stat-label">Accounts</div><div class="stat-value">${users.length}</div></div>
      <div class="stat-card tone-green"><div class="stat-label">Active Accounts</div><div class="stat-value">${users.filter(u => u.is_active).length}</div></div>
      <div class="stat-card"><div class="stat-label">Admins</div><div class="stat-value">${users.filter(u => u.role === 'ADMIN').length}</div></div>
      <div class="stat-card tone-amber"><div class="stat-label">Auth Events Logged</div><div class="stat-value">${authEvents.length}</div></div>
    </div>
    <div class="card">
      <div class="panel-title"><h3>Sensitive Events</h3><span class="sub">Auth, evidence access and case deletion</span></div>
      <div style="display:flex;flex-direction:column;gap:2px;">
        ${sensitive.length ? sensitive.slice(0, 40).map(l => `
          <div style="display:flex;justify-content:space-between;gap:10px;padding:12px 0;border-bottom:1px solid var(--border-soft);font-size:13px;">
            <span><span class="pill pill-blue" style="margin-right:8px;">${escapeHtml(l.type)}</span>${escapeHtml(l.eventType)} — <span class="mono" style="color:var(--text-dim);">${escapeHtml(l.actor)}</span></span>
            <span class="mono" style="color:var(--text-dim);">${fmtDate(l.time)}</span>
          </div>`).join('') : `<p style="color:var(--text-muted);font-size:13px;margin:0;">No security-relevant events recorded yet.</p>`}
      </div>
    </div>
    <div class="disclaimer-box" style="margin-top:18px;">${ic('helpCircle')}
      Authentication, authorization and audit logging are enforced server-side. Failed-login attempts are not
      persisted as audit rows, so no failed-login count can be shown here.
    </div>
  `;
  return appShell('admin', '#/admin-security', 'Security', inner);
}

/* Read-only view of the deployment the frontend is actually talking to. There
   is no settings endpoint, so nothing here is editable or invented. */
async function renderAdminSettings() {
  let me = null;
  try { me = await MayaApi.auth.me(); } catch (err) { /* shown as unavailable */ }
  const inner = `
    <p class="page-sub">Runtime information reported by the backend. No writable settings endpoint exists.</p>
    <div class="card" style="max-width:620px;">
      <div class="panel-title"><h3>Deployment</h3></div>
      <div class="kv-grid">
        <div class="kv-item"><div class="kl">API base URL</div><div class="kv-val mono">${escapeHtml(MayaApi.baseUrl || location.origin)}</div></div>
        <div class="kv-item"><div class="kl">Signed in as</div><div class="kv-val">${escapeHtml(me ? me.username : 'unknown')}</div></div>
        <div class="kv-item"><div class="kl">Role</div><div class="kv-val">${escapeHtml(me ? me.role : 'unknown')}</div></div>
        <div class="kv-item"><div class="kl">Session</div><div class="kv-val">Server-side session cookie</div></div>
      </div>
      <div class="divider"></div>
      <p style="color:var(--text-muted);font-size:12.5px;margin:0;">
        Detection thresholds, storage paths and model configuration are set through backend environment
        configuration and are not exposed over HTTP.
      </p>
    </div>
  `;
  return appShell('admin', '#/admin-settings', 'Settings', inner);
}

/* ==========================================================================
   23. GLOBAL ACTIONS  (event delegation for data-action attributes)
   ========================================================================== */
document.addEventListener('click', (e) => {
  const el = e.target.closest('[data-action]');
  if (!el) return;
  const action = el.dataset.action;

  if (action === 'generate-report') {
    e.preventDefault();
    handleGenerateReport(el);
    return;
  }

  if (action === 'download-report') {
    e.preventDefault();
    downloadReportPdf(el.dataset.report, el.dataset.number);
    return;
  }

  if (action === 'verify-integrity') {
    e.preventDefault();
    handleVerifyIntegrity(el);
    return;
  }

  if (action === 'face-verify') {
    e.preventDefault();
    openFaceVerificationModal(el.dataset.evidence, el.dataset.investigation);
    return;
  }

  if (action === 'logout') {
    e.preventDefault();
    confirmModal('Log out?', 'You will need to sign in again to access your workspace.', async () => {
      // Invalidate the server-side session; the local cache is cleared either
      // way so the UI never shows a signed-in state without a valid cookie.
      try {
        await MayaApi.auth.logout();
        toast('Logged out', 'Your MAYA session has been ended.', 'success');
      } catch (err) {
        if (err instanceof MayaApi.ApiError && err.isAuth) {
          toast('Logged out', 'Your session had already expired.', 'info');
        } else {
          toast('Logout incomplete', err.message || 'Could not reach the backend.', 'error', 6000);
        }
      } finally {
        clearAuth();
        location.hash = '#/home';
      }
    }, 'Logout', false);
    return;
  }

  if (action === 'toggle-sidebar') {
    e.preventDefault();
    $('#sidebar')?.classList.toggle('open');
    document.body.classList.toggle('sidebar-open');
    return;
  }

  if (action === 'download-public-result') {
    e.preventDefault();
    const ctx = getCtx();
    const ev = ctx.pendingEvidence;
    if (!ev) return;
    const txt = [
      'EVIDEX — PUBLIC VERIFICATION RESULT', '=====================================',
      `Filename: ${ev.filename}`, `Type: ${ev.type}`, `Size: ${fmtBytes(ev.size)}`,
      `Checked on: ${fmtDate(new Date())}`, `SHA-256: ${ev.sha256}`,
      '', `Result: ${ev.verdict}`, `Authenticity score: ${ev.score}%`,
      '', 'This is a basic public check. For legal or forensic investigation, sign in as an Investigator.',
      'This is a simulated demo export generated by the EVIDEX frontend prototype.',
    ].join('\n');
    downloadTextFile(`EVIDEX_Result_${(ev.filename||'file').replace(/\.[^.]+$/,'')}.txt`, txt);
    toast('Download started', 'Verification result saved.', 'success');
  }
});

/* Close mobile sidebar when a nav link is tapped or the dark overlay is tapped */
document.addEventListener('click', (e) => {
  if (window.innerWidth > 860) return;
  const link = e.target.closest('.side-link');
  if (link && !link.hasAttribute('data-action')) {
    $('#sidebar')?.classList.remove('open');
    document.body.classList.remove('sidebar-open');
  }
  if (document.body.classList.contains('sidebar-open') && !e.target.closest('.sidebar') && !e.target.closest('[data-action="toggle-sidebar"]')) {
    $('#sidebar')?.classList.remove('open');
    document.body.classList.remove('sidebar-open');
  }
});

/* Close modal on Escape */
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });

/* ==========================================================================
   24. INIT
   ========================================================================== */
/* ==========================================================================
   MAYA-BACKED ACTIONS (report / integrity / face verification)
   ========================================================================== */

async function handleGenerateReport(el) {
  if (el.disabled) return;
  const analysisId = el.dataset.analysis;
  if (!analysisId || analysisId === 'null') {
    toast('No analysis', 'A completed analysis is required before generating a report.', 'error');
    return;
  }
  const original = el.innerHTML;
  el.disabled = true;
  el.innerHTML = 'Generating…';
  try {
    const report = await MayaApi.analysis.generateReport(analysisId);
    const r = MayaApi.adapt.report(report);
    const emailNote = r.emailMessage || 'Report generated.';
    toast(r.emailStatus === 'sent' ? 'Report emailed' : 'Report generated', emailNote, r.emailStatus === 'sent' ? 'success' : 'info', 6000);
    openModal(`
      <h3>${ic('reports')} Report ready</h3>
      <p style="color:var(--text-muted);font-size:13.5px;margin-top:8px;">
        ${escapeHtml(r.reportNumber)} — ${escapeHtml((r.format || 'pdf').toUpperCase())}, ${fmtBytes(r.sizeBytes || 0)}
      </p>
      <p style="color:var(--text-muted);font-size:13.5px;margin-top:8px;">${escapeHtml(emailNote)}</p>
      ${r.emailStatus === 'sent' && r.maskedRecipient ? `<p style="font-size:13px;margin-top:4px;">Sent to ${escapeHtml(r.maskedRecipient)}</p>` : ''}
      <div class="hash-box" style="margin-top:10px;"><span>${escapeHtml(r.sha256 || '')}</span></div>
      <div class="modal-actions">
        <button class="btn btn-primary btn-sm" data-action="download-report" data-report="${r.reportId}" data-number="${escapeHtml(r.reportNumber)}">${ic('download')} Download PDF</button>
        <button class="btn btn-ghost btn-sm" data-close-modal>Close</button>
      </div>`);
    $$('[data-close-modal]').forEach(x => x.addEventListener('click', closeModal));
  } catch (err) {
    handleApiError(err, 'Report generation failed');
  } finally {
    el.disabled = false;
    el.innerHTML = original;
  }
}

/* The download endpoint returns a real PDF stream. Fetch it with the session
   cookie, then save the blob — never parse it as JSON. */
async function downloadReportPdf(reportId, reportNumber) {
  toast('Preparing download', 'Fetching the PDF from MAYA…', 'info', 2000);
  try {
    const res = await fetch(MayaApi.reports.downloadUrl(reportId), { credentials: 'include' });
    if (!res.ok) {
      // Errors still arrive as the JSON envelope; surface the real status.
      let message = `Download failed (${res.status}).`;
      try {
        const body = await res.json();
        if (body && body.message) message = body.message;
      } catch (parseErr) { /* keep the status-based message */ }
      throw new MayaApi.ApiError(res.status, 'download_failed', message);
    }
    const blob = await res.blob();
    const disposition = res.headers.get('Content-Disposition') || '';
    const match = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(disposition);
    const filename = match ? decodeURIComponent(match[1]) : `${reportNumber || 'report'}.pdf`;

    const objectUrl = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = objectUrl;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(objectUrl), 10000);
    toast('Download started', filename, 'success');
  } catch (err) {
    handleApiError(err, 'Download failed');
  }
}

async function handleVerifyIntegrity(el) {
  const evidenceId = el.dataset.evidence;
  const original = el.innerHTML;
  el.disabled = true;
  el.innerHTML = 'Verifying…';
  try {
    const result = await MayaApi.evidence.verifyIntegrity(evidenceId);
    const ok = result.match !== false && result.integrity_verified !== false;
    toast(ok ? 'Integrity intact' : 'Integrity MISMATCH',
      ok ? 'The stored file still matches its ingest SHA-256.'
         : 'The file no longer matches the hash recorded at ingest.',
      ok ? 'success' : 'error', 7000);
  } catch (err) {
    handleApiError(err, 'Integrity check failed');
  } finally {
    el.disabled = false;
    el.innerHTML = original;
  }
}

/* Phase 5 face verification, reused as-is. The reference image is posted to
   the existing endpoint and the stored decision is shown verbatim. */
function openFaceVerificationModal(evidenceId, investigationId) {
  openModal(`
    <h3>${ic('investigators')} Face Reference Verification</h3>
    <p style="color:var(--text-muted);font-size:13.5px;margin-top:8px;">
      Upload a reference face image. It is compared against the face detected in evidence EV-${escapeHtml(String(evidenceId))}.
    </p>
    <div class="field" style="margin-top:12px;">
      <label for="faceRefInput">Reference image</label>
      <input type="file" id="faceRefInput" accept=".jpg,.jpeg,.png,.bmp,.webp,image/*">
    </div>
    <div class="field">
      <label for="faceThreshold">Match threshold (optional)</label>
      <input type="number" id="faceThreshold" step="0.01" min="0" max="1" placeholder="Backend default">
    </div>
    <div id="faceResult"></div>
    <div class="modal-actions">
      <button class="btn btn-primary btn-sm" id="faceRunBtn">${ic('scan')} Run verification</button>
      <button class="btn btn-ghost btn-sm" data-close-modal>Close</button>
    </div>`);
  $$('[data-close-modal]').forEach(x => x.addEventListener('click', closeModal));

  $('#faceRunBtn').addEventListener('click', async () => {
    const input = $('#faceRefInput');
    const file = input.files && input.files[0];
    if (!file) { toast('No file', 'Select a reference image first.', 'error'); return; }
    const btn = $('#faceRunBtn');
    const thresholdRaw = $('#faceThreshold').value;
    btn.disabled = true;
    btn.innerHTML = 'Verifying…';
    $('#faceResult').innerHTML = '';
    try {
      const row = await MayaApi.faceVerification.create(evidenceId, file, {
        threshold: thresholdRaw === '' ? null : Number(thresholdRaw),
        investigationId: investigationId || null,
      });
      const decision = row.decision || row.verification_status;
      $('#faceResult').innerHTML = `<div style="margin-top:14px;">${faceVerificationResultHtml(row)}</div>`;
      toast('Face verification complete', String(decision), decision === 'MATCH' ? 'success' : 'info', 5000);
    } catch (err) {
      handleApiError(err, 'Face verification failed');
    } finally {
      btn.disabled = false;
      btn.innerHTML = `${ic('scan')} Run verification`;
    }
  });
}

async function initApp() {
  getDB(); // seeded dataset now only backs the public verification demo page
  // Re-establish the real session before the first render so a valid cookie
  // survives a page reload (and a stale local cache never fakes a login).
  if (getAuth()) await syncAuth();
  router();
}

document.addEventListener('DOMContentLoaded', initApp);
