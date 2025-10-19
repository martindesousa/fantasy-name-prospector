// Main form
const form = document.getElementById('name-generator-form');

// References to the main form fields
const prefixInput = document.getElementById('prefix');
const lengthInput = document.getElementById('length');
const lengthModeSelect = document.getElementById('length_mode');
const errorMessage = document.getElementById('error-message');
// preserve default error HTML so we can restore it after custom validation messages
const defaultErrorHTML = errorMessage ? errorMessage.innerHTML : '';

// References to loading elements
const loadingDiv = document.getElementById('loading');
const loadingText = document.getElementById('loading-text');
const progressBar = document.getElementById('progress-bar');

// Reference to the inline spinner (Swap this out on errors)
let spinnerEl = loadingDiv ? loadingDiv.querySelector('.spinner-border') : null;

// Reference to generate and cancel buttons
const generateButton = document.getElementById('generate-button');
const cancelButton = document.getElementById('cancel-generate-button');

// Track generation state
let isGenerating = false;
let abortController = null;

// Event delegation for name item clicks - one listener for name items
// Attached to container-fluid so it works even when name-grid doesn't exist yet
document.querySelector('.container-fluid').addEventListener('click', function(e) {
    // Check if the clicked element is a name-item (or inside one)
    const nameItem = e.target.closest('.name-item');
    if (nameItem) {
        const name = nameItem.textContent;
        navigator.clipboard.writeText(name).then(() => {
            // Store original background for restoration
            const originalBg = nameItem.style.backgroundColor || '';
            nameItem.style.backgroundColor = '#28a745';
            setTimeout(() => {
                nameItem.style.backgroundColor = originalBg;
            }, 200);
            // show transient copied notification
            showCopiedToast('Copied to clipboard');
        }).catch(err => {
            console.warn('Failed to copy to clipboard:', err);
        });
    }
});


// References to the custom form fields
const modelSelect = document.getElementById('model');
const customNamesContainer = document.getElementById('custom-names-input') || document.getElementById('custom-names-container');
const customNotice = document.getElementById('custom-notice');

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
            // remove numeric constraints when not numeric
            lengthInput.removeAttribute('min');
            lengthInput.removeAttribute('max');
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
            lengthInput.placeholder = '0 - 20';
            lengthInput.disabled = false;
            lengthInput.name = 'length';
            // enforce reasonable bounds for custom length
            lengthInput.min = 1;
            lengthInput.max = 20;
        }
    });

    // Trigger initial length-mode update
    lengthModeSelect.dispatchEvent(new Event('change'));
}

// Cancel button handler
if (cancelButton) {
    cancelButton.addEventListener('click', function() {
        if (abortController) {
            abortController.abort();
            loadingText.textContent = 'Generation cancelled';
            showWarningImage();
            
            // Hide cancel button and loading after a brief delay
            setTimeout(() => {
                loadingDiv.style.display = 'none';
                isGenerating = false;
                generateButton.disabled = false;
                cancelButton.style.display = 'none';
            }, 1500);
        }
    });
}

// Form submit handler with streaming progress
form.addEventListener('submit', function(event) {
    event.preventDefault();

    // Prevent multiple simultaneous generations
    if (isGenerating) {
        return;
    }

    const prefixText = prefixInput.value;
    const lengthMode = lengthModeSelect ? lengthModeSelect.value : 'average';
    const nameLength = lengthMode === 'custom' ? parseInt(lengthInput.value) : null;

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

    // Additional validation: enforce maximum allowed custom length
    if (lengthMode === 'custom' && Number.isFinite(nameLength)) {
        if (nameLength > 20) {
            errorMessage.style.display = 'block';
            errorMessage.innerHTML = '<i class="bi bi-exclamation-triangle"></i> Error: Maximum allowed length is 20.';
            lengthInput.classList.add('input-error');
            return;
        } else {
            // restore default message if previously modified
            if (errorMessage && defaultErrorHTML) errorMessage.innerHTML = defaultErrorHTML;
        }
    }

    // Set generation state and disable generate button
    isGenerating = true;
    generateButton.disabled = true;

    // Create new AbortController for this request
    abortController = new AbortController();

    loadingDiv.style.display = 'block';
    loadingText.textContent = 'Preparing...';
    progressBar.style.width = '0%';
    cancelButton.style.display = 'inline-block';
    
    // ensure spinner visible (in case previous action showed the warning image)
    restoreSpinner();

    // If using a custom model, mark it as used now (so it moves to top only when actually used)
    try {
        const modelType = document.getElementById('model-type') ? document.getElementById('model-type').value : null;
        const selectedHidden = document.getElementById('selected-custom-model');
        const selId = selectedHidden ? selectedHidden.value : null;
        if (modelType === 'custom' && selId && typeof markModelUsed === 'function') {
            try { markModelUsed(selId); } catch(e) { console.warn('markModelUsed failed', e); }
        }
    } catch(e) {}

    // Clear previous results using the centralized results module
    if (typeof results !== 'undefined' && results.clear) {
        try { results.clear(); } catch (e) { console.warn('results.clear failed', e); }
    }

    const formData = new FormData(form);

    fetch('/stream_progress', {
        method: 'POST',
        body: formData,
        signal: abortController.signal
    }).then(response => {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        function processStream() {
            return reader.read().then(({ done, value }) => {
                if (done) {
                    console.log('Stream complete');
                    // Re-enable generation
                    isGenerating = false;
                    generateButton.disabled = false;
                    cancelButton.style.display = 'none';
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

                                case 'generating':
                                    loadingDiv.style.display = 'block';
                                    restoreSpinner();

                                    if (jsonData.name) {
                                        try {
                                            if (typeof results !== 'undefined' && results.addName) {
                                                results.addName(jsonData.name);
                                            }
                                        } catch (e) {
                                            console.warn('results.addName failed', e);
                                        }
                                    }
                                    break;

                                case 'training_complete':
                                    loadingText.textContent = "Training complete! Starting name generation...";
                                    // make sure spinner is present when moving to generation
                                    restoreSpinner();
                                    break;

                                case 'complete':
                                    loadingDiv.style.display = 'none';
                                    isGenerating = false;
                                    generateButton.disabled = false;
                                    cancelButton.style.display = 'none';

                                    // Remove any temporary spacer we added earlier
                                    const spacerFinal = document.getElementById('results-spacer');
                                    if (spacerFinal) spacerFinal.remove();

                                    if (jsonData.names && Array.isArray(jsonData.names)) {
                                        try {
                                            if (typeof results !== 'undefined' && results.renderFinal) {
                                                results.renderFinal(jsonData.names);
                                            }
                                        } catch (e) {
                                            console.warn('results.renderFinal failed', e);
                                        }
                                    }
                                    break;

                                case 'error':
                                    loadingText.textContent = "Error: " + jsonData.message;
                                    loadingDiv.style.display = 'block';
                                    isGenerating = false;
                                    generateButton.disabled = false;
                                    cancelButton.style.display = 'none';
                                    // replace spinner with warning image
                                    showWarningImage();
                                    break;
                            }
                        } catch (e) {
                            console.error('Error parsing SSE data:', e);
                            // show warning image for parsing/stream issues
                            loadingText.textContent = 'Error: stream parsing failed';
                            isGenerating = false;
                            generateButton.disabled = false;
                            cancelButton.style.display = 'none';
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
        
        // Check if this was an abort (user cancelled)
        if (error.name === 'AbortError') {
            console.log('Request was cancelled by user');
            // Don't show error message for user-initiated cancellations
            // The cancel button handler already updated the UI
            return;
        }
        
        loadingText.textContent = 'Error: ' + error.message;
        isGenerating = false;
        generateButton.disabled = false;
        cancelButton.style.display = 'none';
        showWarningImage();
    });
});


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