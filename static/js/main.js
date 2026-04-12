/* Eye Disease Detection – Main JavaScript */

'use strict';

// ── Bootstrap tooltip init ─────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el =>
    new bootstrap.Tooltip(el)
  );

  // Auto-dismiss alerts after 5s
  document.querySelectorAll('.alert.alert-dismissible').forEach(el => {
    setTimeout(() => {
      const bsAlert = bootstrap.Alert.getOrCreateInstance(el);
      if (bsAlert) bsAlert.close();
    }, 5000);
  });
});
