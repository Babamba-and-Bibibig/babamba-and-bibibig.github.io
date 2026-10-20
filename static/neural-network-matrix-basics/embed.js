/* Resize only a marked, same-origin matrix lesson iframe that sent the message. */
(() => {
  'use strict';
  window.addEventListener('message', event => {
    if (event.origin !== window.location.origin || !event.data || event.data.type !== 'matrix-lab-height') return;
    const height = event.data.height;
    if (!Number.isFinite(height) || height < 250 || height > 5000) return;
    document.querySelectorAll('iframe[data-neural-demo]').forEach(frame => {
      if (event.source === frame.contentWindow) frame.style.height = `${Math.ceil(height)}px`;
    });
  });
})();
