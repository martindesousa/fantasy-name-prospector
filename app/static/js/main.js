// Extracted JS from index.html for readability. Keep behavior identical.

// Main form
const form = document.getElementById('name-generator-form');

// References to the form fields for dynamic error message
const prefixInput = document.getElementById('prefix');
const lengthInput = document.getElementById('length');
const lengthModeSelect = document.getElementById('length_mode');
const errorMessage = document.getElementById('error-message');

// References to loading elements
const loadingDiv = document.getElementById('loading');
const loadingText = document.getElementById('loading-text');
const progressBar = document.getElementById('progress-bar');

// References to the form fields for dynamic type box
const modelSelect = document.getElementById('model');
const customNamesContainer = document.getElementById('custom-names-input') || document.getElementById('custom-names-container');
const customNotice = document.getElementById('custom-notice');
const customNamesText = document.getElementById('custom_names');

const trainButton = document.getElementById('train-button');
const generateButton = document.getElementById('generate-button');

// Track the last custom text that was successfully trained
let lastTrainedCustomText = '';

function updateLengthPlaceholder() {
    const visualMode = document.getElementById('model-type') ? document.getElementById('model-type').value : null;
    const selectedModel = visualMode === 'custom' ? 'custom' : modelSelect.value;

    if (selectedModel === 'custom') {
        lengthInput.placeholder = 'default: 6';
        return;
    }

    fetch(`/get_model_avg_length?model=${selectedModel}`)
        .then(response => response.json())
        .then(data => {
            const prefix = data.is_default ? 'default: ' : 'average: ';
            lengthInput.placeholder = prefix + data.avg_length;
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

// Enable/disable length input based on mode
if (lengthModeSelect) {
    lengthModeSelect.addEventListener('change', function() {
        if (this.value === 'custom') {
            lengthInput.disabled = false;
        } else {
            lengthInput.disabled = true;
        }
    });
}

// Function to update button visibility based on model and training status
function updateButtonVisibility() {
    if (modelSelect.value === 'custom') {
        const currentCustomText = customNamesText.value.trim();

        if (!currentCustomText) {
            trainButton.style.display = 'inline-block';
            trainButton.disabled = true;
            generateButton.style.display = 'none';
        } else if (currentCustomText === lastTrainedCustomText && lastTrainedCustomText !== '') {
            trainButton.style.display = 'none';
            generateButton.style.display = 'inline-block';
            generateButton.disabled = false;
        } else {
            trainButton.disabled = false;
            checkIfModelExistsDebounced();
        }
    } else {
        trainButton.style.display = 'none';
        generateButton.style.display = 'inline-block';
        generateButton.disabled = false;
    }
}

// Function to toggle the visibility of the custom names input field
modelSelect.addEventListener('change', function() {
    if (modelSelect.value === 'custom') {
        if (customNamesContainer) customNamesContainer.style.display = 'block';
        if (customNotice) customNotice.style.display = 'block';
    } else {
        if (customNamesContainer) customNamesContainer.style.display = 'none';
        if (customNotice) customNotice.style.display = 'none';
    }
    updateButtonVisibility();
});

// Listen for changes in custom names text
if (customNamesText) customNamesText.addEventListener('input', function() {
    updateButtonVisibility();
});

// Trigger the change event to ensure the correct initial state
modelSelect.dispatchEvent(new Event('change'));

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

    const existingResults = document.querySelector('.results-section');
    if (existingResults) existingResults.remove();

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
                                    break;

                                case 'generating':
                                    loadingDiv.style.display = 'block';

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
                                    if (modelSelect.value === 'custom') {
                                        lastTrainedCustomText = customNamesText.value.trim();
                                        trainButton.style.display = 'none';
                                        generateButton.style.display = 'inline-block';
                                    }
                                    break;

                                case 'complete':
                                    loadingDiv.style.display = 'none';

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

                                    if (modelSelect.value === 'custom' && trainButton.style.display !== 'none') {
                                        lastTrainedCustomText = customNamesText.value.trim();
                                        updateButtonVisibility();
                                    }
                                    break;

                                case 'error':
                                    loadingText.textContent = "Error: " + jsonData.message;
                                    loadingDiv.style.display = 'block';
                                    break;
                            }
                        } catch (e) {
                            console.error('Error parsing SSE data:', e);
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
    });
});

// Function for finding MD5 hash
function hashCustomNames(text) {
    const hashHex = md5(text);
    return `custom_${hashHex}`;
}

let debounceTimeout = null;

function checkIfModelExistsDebounced() {
    console.log("Running check for model existence");
    if (debounceTimeout) {
        clearTimeout(debounceTimeout);
    }

    debounceTimeout = setTimeout(() => {
        const model = modelSelect.value;
        const customNamesTextValue = customNamesText ? customNamesText.value.trim() : '';

        if (model !== 'custom' || !customNamesTextValue) {
            return;
        }

        const hashedModelName = hashCustomNames(customNamesTextValue);
        console.log("Expected model filename:", hashedModelName);

        fetch('/check_model_exists?model=' + encodeURIComponent(hashedModelName))
            .then(response => response.json())
            .then(data => {
                const modelExists = data.exists;

                if (modelExists) {
                    lastTrainedCustomText = customNamesTextValue;
                    trainButton.style.display = 'none';
                    generateButton.style.display = 'inline-block';
                } else {
                    trainButton.style.display = 'inline-block';
                    generateButton.style.display = 'none';
                }
            })
            .catch(err => {
                console.error("Error checking model:", err);
                trainButton.style.display = 'inline-block';
                generateButton.style.display = 'none';
            });
    }, 500);
}

// Run initial check
checkIfModelExistsDebounced();

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

            // If the select has an option with value 'custom', set it. Otherwise create a hidden input
            const hasCustomOption = Array.from(modelSelectEl.options).some(o => o.value === 'custom');
            if (hasCustomOption) {
                modelSelectEl.value = 'custom';
            } else {
                // remove name from select so it doesn't submit
                modelSelectEl.removeAttribute('name');
                // add hidden input to submit model=custom
                let hidden = document.getElementById('model-hidden');
                if (!hidden) {
                    hidden = document.createElement('input');
                    hidden.type = 'hidden';
                    hidden.id = 'model-hidden';
                    hidden.name = 'model';
                    hidden.value = 'custom';
                    form.appendChild(hidden);
                } else {
                    hidden.value = 'custom';
                }
            }

            if (customNamesContainer) customNamesContainer.style.display = 'block';
            if (customNotice) customNotice.style.display = 'block';
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
            // Ensure action buttons reflect template mode: show generate, hide train
            trainButton.style.display = 'none';
            generateButton.style.display = 'inline-block';
            generateButton.disabled = false;
            // Update model-dependent UI state
            if (modelSelectEl.value === 'custom') {
                checkIfModelExistsDebounced();
            } else {
                updateLengthPlaceholder();
            }
        }
    });
});

// Ensure custom tab shows textarea if model is custom on load
if (document.getElementById('model').value === 'custom') {
    const customBtn = Array.from(toggleButtons).find(b => b.getAttribute('data-type') === 'custom');
    if (customBtn) customBtn.click();
}
