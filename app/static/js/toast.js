// Create and show a temporary "Copied" toast message
function showCopiedToast(message, timeout=1400) {
    try {
        const existing = document.getElementById('copied-toast');
        if (existing) existing.remove();

        const t = document.createElement('div');
        t.id = 'copied-toast';
        t.className = 'copied-toast';
        t.textContent = message;
        // Accessibility: polite announcement for screen readers
        t.setAttribute('role', 'status');
        t.setAttribute('aria-live', 'polite');
        document.body.appendChild(t);
        // Trigger reflow then show
        void t.offsetWidth;
        t.classList.add('show');

        setTimeout(() => {
            t.classList.remove('show');
            setTimeout(() => { try { t.remove(); } catch(e){} }, 220);
        }, timeout);
    } catch (e) {
        console.warn('Toast failed', e);
    }
}

window.showCopiedToast = showCopiedToast;
