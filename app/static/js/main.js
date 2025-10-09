// Main form
const form = document.getElementById('name-generator-form');

// References to the main form fields
const prefixInput = document.getElementById('prefix');
const lengthInput = document.getElementById('length');
const lengthModeSelect = document.getElementById('length_mode');
const errorMessage = document.getElementById('error-message');

// References to loading elements
const loadingDiv = document.getElementById('loading');
const loadingText = document.getElementById('loading-text');
const progressBar = document.getElementById('progress-bar');

// Reference to the inline spinner (Swap this out on errors)
let spinnerEl = loadingDiv ? loadingDiv.querySelector('.spinner-border') : null;

// Create a fresh spinner element (so we can restore it after showing the warning image)
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

// Function to show the warning image in place of the spinner
function showWarningImage() {
    if (!loadingDiv) return;
    // avoid duplicating
    if (loadingDiv.querySelector('#loading-warning-image')) return;

    // remove any existing spinner
    const existingSpinner = loadingDiv.querySelector('.spinner-border');
    if (existingSpinner) existingSpinner.remove();

    // create img
    const img = document.createElement('img');
    img.id = 'loading-warning-image';
    img.src = '/static/images/WarningSign.webp';
    img.alt = 'Warning';
    // size similar to spinner
    img.style.width = '2rem';
    img.style.height = '2rem';
    img.style.objectFit = 'contain';
    img.style.marginRight = '0.5rem';

    // insert before the loading text
    const textNode = loadingDiv.querySelector('#loading-text');
    if (textNode) textNode.parentNode.insertBefore(img, textNode);
}

// Function to restore spinner (remove warning image if present and ensure spinner exists)
function restoreSpinner() {
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

// References to the custom form fields
const modelSelect = document.getElementById('model');
const customNamesContainer = document.getElementById('custom-names-input') || document.getElementById('custom-names-container');
const customNotice = document.getElementById('custom-notice');
const customNamesText = document.getElementById('custom_names');

const trainButton = document.getElementById('train-button');
const generateButton = document.getElementById('generate-button');

// Track the last custom text that was successfully trained
let lastTrainedCustomText = '';

function updateLengthPlaceholder() {
    const modelTypeEl = document.getElementById('model-type');
    const visualMode = modelTypeEl ? modelTypeEl.value : null;
    // Prefer concrete hidden 'model' value when available (set by custom_models.js)
    const modelHidden = document.getElementById('model-hidden');
    const selectedModel = modelHidden && modelHidden.value ? modelHidden.value : (visualMode === 'custom' ? null : modelSelect.value);
    // Only update the placeholder when the user has selected Average mode.
    const mode = lengthModeSelect ? lengthModeSelect.value : null;
    if (mode !== 'average') {
        return;
    }

    if (!selectedModel) {
        // no concrete model id to query (custom tab open but nothing selected)
        lengthInput.placeholder = 'default: 6';
        return;
    }

    fetch(`/get_model_avg_length?model=${encodeURIComponent(selectedModel)}`)
        .then(response => response.json())
        .then(data => {
            const prefix = data.is_default ? 'default: ' : 'average: ';
            // Use placeholder so we don't overwrite user's field when in auto/custom
            lengthInput.placeholder = prefix + data.avg_length;
            // Clear any accidental value
            if (lengthInput.type !== 'number') lengthInput.value = '';
        })
        .catch(error => {
            console.log('Error fetching avg_length:', error);
            lengthInput.placeholder = 'default: 6';
        });
}

// Add event listener for model selection changes
modelSelect.addEventListener('change', function() {
    updateLengthPlaceholder();
});

// Update placeholder on page load
updateLengthPlaceholder();

// Enable/disable length input based on mode - unified textbox
if (lengthModeSelect) {
    lengthModeSelect.addEventListener('change', function() {
        const mode = this.value;
        if (mode === 'auto') {
            lengthInput.type = 'text';
            lengthInput.value = '';
            lengthInput.placeholder = 'automatic';
            lengthInput.disabled = true;
            lengthInput.removeAttribute('name');
        } else if (mode === 'average') {
            lengthInput.type = 'text';
            lengthInput.value = '';
            lengthInput.disabled = true;
            lengthInput.removeAttribute('name');
            const modelTypeEl = document.getElementById('model-type');
            const visualMode = modelTypeEl ? modelTypeEl.value : null;
            const modelHidden = document.getElementById('model-hidden');
            const selectedModel = modelHidden && modelHidden.value ? modelHidden.value : (visualMode === 'custom' ? null : modelSelect.value);
            if (!selectedModel) {
                lengthInput.placeholder = 'average: 6';
                return;
            }

            fetch(`/get_model_avg_length?model=${encodeURIComponent(selectedModel)}`)
                .then(r => r.json())
                .then(data => {
                    const prefix = data.is_default ? 'default: ' : 'average: ';
                    lengthInput.placeholder = prefix + data.avg_length;
                })
                .catch(err => {
                    lengthInput.placeholder = 'average: 21';
                });
        } else if (mode === 'custom') {
            lengthInput.type = 'number';
            lengthInput.value = '';
            lengthInput.placeholder = 'Enter length';
            lengthInput.disabled = false;
            lengthInput.name = 'length';
            lengthInput.min = 1;
        }
    });

    // Trigger initial length-mode update
    lengthModeSelect.dispatchEvent(new Event('change'));
}

// Form submit handler with streaming progress
form.addEventListener('submit', function(event) {
    event.preventDefault();

    const prefixText = prefixInput.value;
    const lengthMode = lengthModeSelect ? lengthModeSelect.value : 'average';
    const nameLength = lengthMode === 'custom' ? parseInt(lengthInput.value) : null;
    const customNamesTextValue = customNamesText ? customNamesText.value.trim() : '';

    if (lengthMode === 'custom' && prefixText.length > nameLength) {
        errorMessage.style.display = 'block';
        prefixInput.classList.add('input-error');
        lengthInput.classList.add('input-error');
        return;
    } else {
        errorMessage.style.display = 'none';
        prefixInput.classList.remove('input-error');
        lengthInput.classList.remove('input-error');
    }

    loadingDiv.style.display = 'block';
    loadingText.textContent = 'Preparing...';
    progressBar.style.width = '0%';
    // ensure spinner visible (in case previous action showed the warning image)
    restoreSpinner();

    const existingResults = document.querySelector('.results-section');
    if (existingResults) existingResults.remove();
    // Remove any leftover spacer from previous runs
    const prevSpacer = document.getElementById('results-spacer');
    if (prevSpacer) prevSpacer.remove();

    const formData = new FormData(form);

    fetch('/stream_progress', {
        method: 'POST',
        body: formData
    }).then(response => {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        function processStream() {
            return reader.read().then(({ done, value }) => {
                if (done) {
                    console.log('Stream complete');
                    return;
                }

                const text = decoder.decode(value);
                const lines = text.split('\n\n');

                lines.forEach(line => {
                    if (line.startsWith('data:')) {
                        try {
                            const jsonData = JSON.parse(line.substring(5).trim());
                            console.log("Received event data:", jsonData);

                            if (jsonData.message) {
                                loadingText.textContent = jsonData.message;
                            }

                            if (jsonData.progress !== undefined) {
                                progressBar.style.width = jsonData.progress + '%';
                            }

                            switch(jsonData.type) {
                                case 'preparing':
                                case 'loading':
                                case 'training':
                                    loadingDiv.style.display = 'block';
                                    // ensure spinner is visible if previously replaced
                                    restoreSpinner();
                                    break;
                                // 'preparing' handled above together with loading/training

                                case 'generating':
                                    loadingDiv.style.display = 'block';
                                    restoreSpinner();

                                    let resultsContainer = document.querySelector('.results-section');
                                    if (!resultsContainer) {
                                        resultsContainer = document.createElement('div');
                                        resultsContainer.className = 'main-container results-section';
                                        resultsContainer.innerHTML = `
                                            <div class="results-header">
                                                <h3>Generated Names</h3>
                                                <span class="badge bg-primary" id="name-count">0 names</span>
                                            </div>
                                            <div class="name-grid" id="generated-name-grid">
                                            </div>
                                        `;
                                        document.querySelector('.container-fluid').appendChild(resultsContainer);
                                        // Insert a temporary spacer at the end of the document so scrollIntoView lands the
                                        // results container more centrally even if it expands after being shown.
                                        try {
                                            let spacer = document.getElementById('results-spacer');
                                            if (!spacer) {
                                                spacer = document.createElement('div');
                                                spacer.id = 'results-spacer';
                                                // tune height as needed
                                                spacer.style.height = '240px';
                                                spacer.style.pointerEvents = 'none';
                                                document.body.appendChild(spacer);
                                            }

                                            // Here, scroll the results into view
                                            resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });

                                            const previousTransition = resultsContainer.style.transition;
                                            resultsContainer.style.transition = 'box-shadow 220ms ease-in-out';
                                            resultsContainer.style.boxShadow = '0 0 0 4px rgba(23, 109, 201, 1)';
                                            setTimeout(() => {
                                                resultsContainer.style.boxShadow = '0 0 0 0 rgba(0,0,0,0)';
                                                setTimeout(() => { resultsContainer.style.transition = previousTransition; }, 240);
                                            }, 360);
                                        } catch (e) {
                                            console.warn('Scroll/flash for results failed', e);
                                        }
                                    }

                                    if (jsonData.name) {
                                        const nameGrid = document.getElementById('generated-name-grid');
                                        const nameItem = document.createElement('div');
                                        nameItem.className = 'name-item';
                                        nameItem.textContent = jsonData.name;

                                        nameItem.addEventListener('click', function() {
                                            navigator.clipboard.writeText(jsonData.name).then(() => {
                                                const originalBg = nameItem.style.backgroundColor;
                                                nameItem.style.backgroundColor = '#28a745';
                                                setTimeout(() => {
                                                    nameItem.style.backgroundColor = originalBg;
                                                }, 200);
                                            });
                                        });

                                        nameGrid.appendChild(nameItem);

                                        const countBadge = document.getElementById('name-count');
                                        if (countBadge) {
                                            const currentCount = nameGrid.children.length;
                                            countBadge.textContent = `${currentCount} names`;
                                        }
                                    }
                                    break;

                                case 'training_complete':
                                    loadingText.textContent = "Training complete! Starting name generation...";
                                    // make sure spinner is present when moving to generation
                                    restoreSpinner();
                                    if (modelSelect.value === 'custom') {
                                        lastTrainedCustomText = customNamesText.value.trim();
                                        trainButton.style.display = 'none';
                                        generateButton.style.display = 'inline-block';
                                    }
                                    break;

                                case 'complete':
                                    loadingDiv.style.display = 'none';

                                    // Remove any temporary spacer we added earlier
                                    const spacerFinal = document.getElementById('results-spacer');
                                    if (spacerFinal) spacerFinal.remove();

                                    let finalResultsContainer = document.querySelector('.results-section');
                                    if (!finalResultsContainer) {
                                        finalResultsContainer = document.createElement('div');
                                        finalResultsContainer.className = 'main-container results-section';
                                        finalResultsContainer.innerHTML = `
                                            <div class="results-header">
                                                <h3>Generated Names</h3>
                                                <span class="badge bg-primary">${jsonData.names.length} names</span>
                                            </div>
                                            <div class="name-grid">
                                                ${jsonData.names.map(name => `<div class="name-item" onclick="navigator.clipboard.writeText('${name}')">${name}</div>`).join('')}
                                            </div>
                                        `;
                                        document.querySelector('.container-fluid').appendChild(finalResultsContainer);
                                    }
                                    break;

                                case 'error':
                                    loadingText.textContent = "Error: " + jsonData.message;
                                    loadingDiv.style.display = 'block';
                                    // replace spinner with warning image
                                    showWarningImage();
                                    break;
                            }
                        } catch (e) {
                            console.error('Error parsing SSE data:', e);
                            // show warning image for parsing/stream issues
                            loadingText.textContent = 'Error: stream parsing failed';
                            showWarningImage();
                        }
                    }
                });

                return processStream();
            });
        }

        return processStream();
    }).catch(error => {
        console.error('Fetch error:', error);
        loadingText.textContent = 'Error: ' + error.message;
        showWarningImage();
    });
});

// --- Toggle UI behavior (merged from new code) ---
const toggleButtons = document.querySelectorAll('.toggle-option');
const tabPanes = document.querySelectorAll('.tab-pane');
const modelTypeInput = document.getElementById('model-type');
const templateTab = document.getElementById('template-tab');
const customTab = document.getElementById('custom-tab');

// Remember last selected template model so we can restore when switching back
let lastTemplateModel = document.getElementById('model').value || 'classic_american';

toggleButtons.forEach(button => {
    button.addEventListener('click', () => {
        // Reset active classes
        toggleButtons.forEach(btn => btn.classList.remove('active'));
        tabPanes.forEach(pane => pane.classList.remove('active'));

        // Activate clicked
        button.classList.add('active');
        const targetId = button.getAttribute('data-target');
        const target = document.getElementById(targetId);
        if (target) target.classList.add('active');

        // Update hidden field
        const type = button.getAttribute('data-type');
        modelTypeInput.value = type;

        // If switched to custom, mark the underlying model select to 'custom'
        const modelSelectEl = document.getElementById('model');
        if (type === 'custom') {
            // store last template model
            lastTemplateModel = modelSelectEl.value;

            // If the select has an option with value 'custom', set it. Otherwise remove the select's name
            // but DO NOT create a placeholder hidden input with value 'custom'. We only want a concrete
            // hidden `model` field when the user explicitly selects a custom model (custom_models.js will
            // create that). Creating a placeholder caused the backend to receive model=custom.
            const hasCustomOption = Array.from(modelSelectEl.options).some(o => o.value === 'custom');
            if (hasCustomOption) {
                modelSelectEl.value = 'custom';
            } else {
                // remove name from select so it doesn't submit; do not create a placeholder hidden input
                modelSelectEl.removeAttribute('name');
                const existingHidden = document.getElementById('model-hidden');
                if (existingHidden) existingHidden.remove();
            }

            if (customNamesContainer) customNamesContainer.style.display = 'block';
            if (customNotice) customNotice.style.display = 'block';

            // If a custom model was already selected (server-rendered or previously chosen),
            // ensure we call the select handler so the concrete hidden `model` input is created.
            try {
                const preselected = document.getElementById('selected-custom-model');
                if (preselected && preselected.value && typeof selectModel === 'function') {
                    // call the shared select handler from custom_models.js
                    selectModel(preselected.value);
                }
            } catch (e) {
                console.warn('Could not auto-select preselected custom model:', e);
            }

        } else {
            // restore template model
            const hidden = document.getElementById('model-hidden');
            if (hidden) {
                hidden.remove();
                modelSelectEl.name = 'model';
            }
            modelSelectEl.value = lastTemplateModel;
            if (modelSelectEl.value !== 'custom') {
                if (customNamesContainer) customNamesContainer.style.display = 'none';
                if (customNotice) customNotice.style.display = 'none';
            }
            generateButton.style.display = 'inline-block';
            generateButton.disabled = false;
            // Update model-dependent UI state
            updateLengthPlaceholder();
        }
    });
});

// // Ensure custom tab shows textarea if model is custom on load
// if (document.getElementById('model').value === 'custom') {
//     const customBtn = Array.from(toggleButtons).find(b => b.getAttribute('data-type') === 'custom');
//     if (customBtn) customBtn.click();
// }

// Initialize Bootstrap tooltips for any elements that use data-bs-toggle="tooltip"
(function initTooltips() {
    try {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.forEach(function (el) {
            // eslint-disable-next-line no-undef
            if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
                new bootstrap.Tooltip(el, { html: true, sanitize: false });
            }
        });
    } catch (e) {
        // non-critical
        console.warn('Tooltip initialization failed:', e);
    }
})();

// Show a small note when temperature is set to 0 (deterministic behavior)
try {
    const tempInput = document.getElementById('temperature');
    const tempNote = document.getElementById('temperature-note');
    if (tempInput && tempNote) {
        const countSelect = document.getElementById('count');
        // store previous count so we can restore when leaving deterministic mode
        let previousCount = countSelect ? countSelect.value : null;

        // Helper to ensure a hidden input named 'count' exists with the given value
        function ensureHiddenCount(value) {
            let hidden = document.getElementById('count-hidden');
            if (!hidden) {
                hidden = document.createElement('input');
                hidden.type = 'hidden';
                hidden.id = 'count-hidden';
                hidden.name = 'count';
                form.appendChild(hidden);
            }
            hidden.value = String(value);
        }

        // Helper to remove the hidden count input
        function removeHiddenCount() {
            const hidden = document.getElementById('count-hidden');
            if (hidden) hidden.remove();
        }

        function updateTempNote() {
            const v = parseFloat(tempInput.value);
            if (!isNaN(v) && v === 0) {
                tempNote.style.display = 'block';
                // Ensure a hidden count is present so disabled selects still submit a value
                ensureHiddenCount('1');
                if (countSelect) {
                    // save previous if not already saved
                    if (!countSelect.dataset._saved) {
                        countSelect.dataset._saved = countSelect.value;
                    }
                    countSelect.value = '1';
                    countSelect.disabled = true;
                }
            } else {
                tempNote.style.display = 'none';
                // Remove the helper hidden input when not in deterministic mode
                removeHiddenCount();
                if (countSelect) {
                    // restore previous value if present
                    if (countSelect.dataset._saved) {
                        countSelect.value = countSelect.dataset._saved;
                        delete countSelect.dataset._saved;
                    }
                    countSelect.disabled = false;
                }
            }
        }
        tempInput.addEventListener('input', updateTempNote);
        // run once on load
        updateTempNote();
    }
} catch (e) {
    console.warn('Temp note init failed', e);
}
