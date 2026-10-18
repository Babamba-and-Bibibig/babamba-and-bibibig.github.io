/* Only resize the matching same-origin, explicitly marked article demo. */
(() => {
  'use strict';
  window.addEventListener('message', event => {
    if (event.origin !== window.location.origin || !event.data || event.data.type !== 'boosting-demo-height') return;
    const height = event.data.height;
    if (!Number.isFinite(height) || height < 200 || height > 16000) return;
    document.querySelectorAll('iframe[data-boosting-demo]').forEach(frame => {
      if (frame.contentWindow === event.source) frame.style.height = `${Math.ceil(height)}px`;
    });
  });
})();
