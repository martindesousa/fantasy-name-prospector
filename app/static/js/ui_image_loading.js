// UI image loading: spinner and warning image
function createSpinner() {
    const s = document.createElement('div');
    s.className = 'spinner-border text-primary me-2';
    s.setAttribute('role', 'status');
    const span = document.createElement('span');
    span.className = 'visually-hidden';
    span.textContent = 'Loading...';
    s.appendChild(span);
    return s;
}

function showWarningImage() {
    const loadingDiv = document.getElementById('loading');
    if (!loadingDiv) return;

    if (loadingDiv.querySelector('#loading-warning-image')) return;
    const existingSpinner = loadingDiv.querySelector('.spinner-border');
    if (existingSpinner) existingSpinner.remove();

    const img = document.createElement('img');
    img.id = 'loading-warning-image';
    img.src = '/static/images/WarningSign.webp';
    img.alt = 'Warning';
    img.style.width = '2rem';
    img.style.height = '2rem';
    img.style.objectFit = 'contain';
    img.style.marginRight = '0.5rem';

    const textNode = loadingDiv.querySelector('#loading-text');
    if (textNode) textNode.parentNode.insertBefore(img, textNode);
}

function restoreSpinner() {
    const loadingDiv = document.getElementById('loading');
    if (!loadingDiv) return;
    const warning = loadingDiv.querySelector('#loading-warning-image');
    if (warning) warning.remove();

    const existingSpinner = loadingDiv.querySelector('.spinner-border');
    if (!existingSpinner) {
        const textNode = loadingDiv.querySelector('#loading-text');
        const newSpinner = createSpinner();
        if (textNode) textNode.parentNode.insertBefore(newSpinner, textNode);
    }
}

// main.js can call these
window.createSpinner = createSpinner;
window.showWarningImage = showWarningImage;
window.restoreSpinner = restoreSpinner;
