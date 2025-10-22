// custom_models.js

// MOCK DATA
let customModels = [
    {
        id: 'fantasy_heroes_001',
        name: 'Fantasy Heroes',
        description: 'Epic fantasy character names',
        category: 'fantasy',
        nameCount: 45,
        createdAt: new Date('2024-01-15').getTime(),
        lastUsed: new Date('2024-01-20').getTime()
    },
    {
        id: 'sci_fi_pilots_002',
        name: 'Sci-Fi Pilots',
        description: 'Futuristic pilot names',
        category: 'sci-fi',
        nameCount: 32,
        createdAt: new Date('2024-01-10').getTime(),
        lastUsed: new Date('2024-01-18').getTime()
    },
    {
        id: 'royal_names_003',
        name: 'Royal Names',
        description: 'Noble and royal names',
        category: 'historical',
        nameCount: 28,
        createdAt: new Date('2024-01-05').getTime(),
        lastUsed: new Date('2024-01-16').getTime()
    }
];

let selectedModelId = null;
let editingModelId = null; // when editing an existing model, this holds its id
let originalTrainingData = null;

// Track training state
let isTraining = false;
let trainingAbortController = null;

// If the server (or previous client state) stored a selected model id in the hidden
// field, initialize our in-memory selection so loadCustomModels can reapply it.
try {
    const preselected = document.getElementById('selected-custom-model');
    if (preselected && preselected.value) {
        selectedModelId = preselected.value;
    }
} catch (e) {
    // non-critical
}

// Called when a custom model is used to generate names (to update its lastUsed timestamp)
function markModelUsed(modelId) {
    const model = customModels.find(m => m.id === modelId);
    if (!model) return;
    model.lastUsed = Date.now();
    renderModelsList();
}

// Reset training state and hide cancel button
function resetTrainingState() {
    isTraining = false;
    const cancelBtn = document.getElementById('cancel-training-button');
    if (cancelBtn) cancelBtn.style.display = 'none';
}

// Initialize the interface
document.addEventListener('DOMContentLoaded', function() {
    // If an initializeInterface function exists in another script, it will run there.
    loadCustomModels();
    setupEventListeners();
});

function setupEventListeners() {
    // Search functionality
    const searchInput = document.getElementById('model-search');
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            filterModels(this.value);
        });
    }

    const createBtn = document.getElementById('create-new-model-btn');
    if (createBtn) createBtn.addEventListener('click', showNewModelForm);

    const saveBtn = document.getElementById('save-custom-model');
    if (saveBtn) saveBtn.addEventListener('click', saveCustomModel);

    const cancelBtn = document.getElementById('cancel-new-model');
    if (cancelBtn) cancelBtn.addEventListener('click', hideNewModelForm);

    const cancelTrainingBtn = document.getElementById('cancel-training-button');
    if (cancelTrainingBtn) {
        cancelTrainingBtn.addEventListener('click', function() {
            if (trainingAbortController) {
                trainingAbortController.abort();
                if (typeof loadingText !== 'undefined' && loadingText) {
                    loadingText.textContent = 'Training cancelled';
                }
                if (typeof showWarningImage === 'function') {
                    showWarningImage();
                }
                
                // Hide cancel button and loading after a brief delay
                setTimeout(() => {
                    if (typeof loadingDiv !== 'undefined' && loadingDiv) {
                        loadingDiv.style.display = 'none';
                    }
                    resetTrainingState();
                }, 1500);
            }
        });
    }

    const form = document.getElementById('name-generator-form');
    if (form) {
        form.addEventListener('submit', function(e) {
            const modelType = document.getElementById('model-type').value;

            if (modelType === 'custom' && !selectedModelId) {
                e.preventDefault();
                showStatusMessage('Please select a custom model or create a new one', 'error');
                return;
            }
            // For streaming endpoint, allow the form submit to continue; main.js handles streaming.
        });
    }
}

// Initialize description and training names resizers and training names count badge
function initResizableTextarea(textareaId, handleId, countBadgeId) {
    const textarea = document.getElementById(textareaId);
    const handle = handleId ? document.getElementById(handleId) : null;
    const countBadge = countBadgeId ? document.getElementById(countBadgeId) : null;

    if (!textarea) return null;

    // Optional count updater (exposed for the custom-names textarea)
    function updateCount() {
        if (!countBadge) return;
        const lines = textarea.value.split('\n').filter(l => l.trim());
        const n = lines.length;
        countBadge.textContent = `${n} ${n === 1 ? 'name' : 'names'}`;
    }

    // Expose update function for the training names textarea specifically
    if (countBadgeId === 'custom-names-count') {
        window.updateCustomNamesCount = updateCount;
    }

    // Initial count update if applicable
    try { updateCount(); } catch (e) {}

    // Update on input events
    textarea.addEventListener('input', function() {
        try { updateCount(); } catch (e) {}
    });

    // Watch for DOM changes that may update the textarea content programmatically
    try {
        const observer = new MutationObserver(function() { try { updateCount(); } catch (e) {} });
        observer.observe(textarea, { characterData: true, childList: true, subtree: true });
    } catch (e) {
        // MutationObserver might not be available - non-fatal
    }

    // If there's no handle supplied, we're done (count-only)
    if (!handle) return { updateCount };

    // Pointer-based smooth drag-to-resize
    let startY = 0;
    let startHeight = 0;
    let dragging = false;

    const onPointerDown = (e) => {
        e.preventDefault();
        dragging = true;
        startY = e.clientY || (e.touches && e.touches[0] && e.touches[0].clientY) || 0;
        startHeight = textarea.getBoundingClientRect().height;
        document.documentElement.style.userSelect = 'none';
        handle.classList.add('active');
        window.addEventListener('pointermove', onPointerMove);
        window.addEventListener('pointerup', onPointerUp);
    };

    const onPointerMove = (e) => {
        if (!dragging) return;
        const clientY = e.clientY || (e.touches && e.touches[0] && e.touches[0].clientY) || 0;
        const dy = clientY - startY;
        const newHeight = Math.max(80, startHeight + dy);
        textarea.style.height = newHeight + 'px';
    };

    const onPointerUp = (e) => {
        dragging = false;
        document.documentElement.style.userSelect = '';
        handle.classList.remove('active');
        window.removeEventListener('pointermove', onPointerMove);
        window.removeEventListener('pointerup', onPointerUp);
        try { updateCount(); } catch (e) {}
    };

    handle.addEventListener('pointerdown', onPointerDown, { passive: false });

    // Keyboard accessibility for the handle
    handle.tabIndex = 0;
    handle.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowUp' || e.key === 'PageUp') {
            textarea.style.height = (textarea.clientHeight - 20) + 'px';
            e.preventDefault();
            try { updateCount(); } catch (err) {}
        } else if (e.key === 'ArrowDown' || e.key === 'PageDown') {
            textarea.style.height = (textarea.clientHeight + 20) + 'px';
            e.preventDefault();
            try { updateCount(); } catch (err) {}
        }
    });

    return { updateCount };
}

// Initialize enhancements when DOM ready: wire both textareas to the shared initializer
document.addEventListener('DOMContentLoaded', function() {
    try {
        // Training names textarea with count badge
        initResizableTextarea('custom-names-input', 'custom-resize-handle', 'custom-names-count');
        // Description textarea (no count badge)
        initResizableTextarea('new-model-description', 'desc-resize-handle', null);
    } catch (e) {
        console.warn('Resizable textarea initialization failed', e);
    }
});

// Function to set the new-model form header text based on if editing or not
function setNewModelHeaderEditing(isEditing, modelId) {
    const header = document.querySelector('#new-model-form h6');
    if (!header) return;
    if (isEditing) {
        header.innerHTML = `<img src="/static/images/SettingsGear.png" alt="Settings" style="width:20px;height:20px;object-fit:contain;margin-right:1px;vertical-align:middle;"> Editing Custom Model`;
    } else {
        header.innerHTML = `<i class="bi bi-plus-circle"></i> Create New Custom Model`;
    }
}

function loadCustomModels() {
    // Load from backend
    fetch('/api/custom_models')
        .then(r => r.json())
        .then(data => {
            if (data && data.models) {
                // normalize model objects but DO NOT include trainingData here (fetch on-demand)
                customModels = data.models.map(m => ({
                    id: m.id,
                    name: m.name || m.id,
                    description: m.description || '',
                    category: m.category || 'uncategorized',
                    nameCount: m.nameCount || 0,
                    createdAt: m.createdAt || Date.now(),
                    lastUsed: m.lastUsed || Date.now()
                }));
            }
            renderModelsList();
            // Reapply selection if previously selected model still exists
            if (selectedModelId) {
                const exists = customModels.some(m => m.id === selectedModelId);
                if (exists) selectModel(selectedModelId);
                else {
                    selectedModelId = null;
                    const hidden = document.getElementById('selected-custom-model'); if (hidden) hidden.value = '';
                    const modelHidden = document.getElementById('model-hidden'); if (modelHidden) modelHidden.remove();
                        // Refresh placeholder if needed
                        try { if (typeof updateLengthPlaceholder === 'function') updateLengthPlaceholder(); } catch(e) {}
                }
            }
        })
        .catch(err => {
            console.error('Error loading custom models:', err);
            renderModelsList();
        });
}

function renderModelsList(filteredModels = null) {
    const modelsList = document.getElementById('custom-models-list');
    const emptyState = document.getElementById('empty-state');
    const modelsToShow = filteredModels || customModels;

    if (!modelsList) return;

    if (modelsToShow.length === 0) {
        modelsList.innerHTML = '';
        if (emptyState) modelsList.appendChild(emptyState);
        return;
    }

    // Sort by last used (most recent first)
    const sortedModels = [...modelsToShow].sort((a, b) => b.lastUsed - a.lastUsed);

    modelsList.innerHTML = sortedModels.map(model => {
        const createdDate = new Date(model.createdAt).toLocaleDateString();
        const lastUsedDate = new Date(model.lastUsed).toLocaleDateString();

        return `
            <div class="custom-model-item d-flex justify-content-between align-items-start ${selectedModelId === model.id ? 'selected' : ''}" 
                data-model-id="${model.id}" 
                style="padding:15px 20px; border-radius:10px;">

                <!-- Info -->
                <div class="model-info flex-grow-1" style="min-width:0; padding-right:15px;">
                <div class="model-name fw-semibold" style="font-size:1.05rem; line-height:1.2; margin-bottom:4px;">
                    ${model.name}
                </div>
                <div class="model-meta text-muted small" style="white-space:normal; word-break:break-word; margin-bottom:6px;">
                    ${model.description || 'No description'}
                </div>
                <div class="model-stats mt-1" style="display:flex; gap:6px; flex-wrap:wrap;">
                    <span class="stat-badge">${model.nameCount} names</span>
                    <span class="stat-badge">${model.category}</span>
                    <span class="stat-badge">Used ${lastUsedDate}</span>
                </div>
                </div>

                <!-- Actions -->
                <div class="model-actions d-flex flex-column align-items-center" style="gap:6px;">
                <button type="button" class="btn btn-outline-secondary btn-sm btn-icon" 
                        data-action="edit" data-id="${model.id}" title="Edit" 
                        style="width:36px;height:36px;display:flex;align-items:center;justify-content:center;padding:0;border-radius:6px;">
                    <img src="/static/images/SettingsGear.png" alt="Edit" style="width:22px;height:22px;object-fit:contain;">
                </button>
                <button type="button" class="btn btn-outline-danger btn-sm btn-icon" 
                        data-action="delete" data-id="${model.id}" title="Delete" 
                        style="width:36px;height:36px;display:flex;align-items:center;justify-content:center;padding:0;border-radius:6px;">
                    <img src="/static/images/TrashCan.png" alt="Delete" style="width:24px;height:24px;object-fit:contain;">
                </button>
                </div>
            </div>
            `;
    }).join('');

    // Attach delegated click listeners
    modelsList.querySelectorAll('.custom-model-item').forEach(item => {
        item.addEventListener('click', function(e) {
            if (!e.target.closest('.model-actions')) {
                selectModel(this.dataset.modelId);
            }
        });
    });

    modelsList.querySelectorAll('.model-actions button').forEach(btn => {
        btn.addEventListener('click', function(e) {
            // Prevent the button from submitting any surrounding form
            e.preventDefault();
            e.stopPropagation();
            const action = this.dataset.action;
            const id = this.dataset.id;
            if (action === 'select') selectModel(id);
            if (action === 'edit') editModel(id);
            if (action === 'delete') deleteModel(id);
        });
    });
}

function filterModels(searchTerm) {
    if (!searchTerm) {
        renderModelsList();
        return;
    }

    const filtered = customModels.filter(model => 
        model.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        (model.description && model.description.toLowerCase().includes(searchTerm.toLowerCase())) ||
        model.category.toLowerCase().includes(searchTerm.toLowerCase())
    );

    renderModelsList(filtered);
}

function selectModel(modelId) {
    selectedModelId = modelId;
    const model = customModels.find(m => m.id === modelId);

    if (!model) return;

    // Update UI display
    const display = document.getElementById('selected-model-display');
    if (display) display.textContent = model.name;

    // Keep a hidden field with the selected id for custom-only endpoints
    const selectedHidden = document.getElementById('selected-custom-model');
    if (selectedHidden) selectedHidden.value = modelId;

    // Ensure a concrete hidden input named 'model' is present so the main form posts the concrete id
    try {
        const form = document.getElementById('name-generator-form');
        if (form) {
            let modelHidden = document.getElementById('model-hidden');
            if (!modelHidden) {
                modelHidden = document.createElement('input');
                modelHidden.type = 'hidden';
                modelHidden.id = 'model-hidden';
                modelHidden.name = 'model';
                form.appendChild(modelHidden);
            }
            modelHidden.value = modelId;
        }
    } catch (e) {
        console.warn('Could not ensure model-hidden input:', e);
    }

    // Re-render to update selection highlight
    renderModelsList();

    // Hide new model form if open
    hideNewModelForm();
        // Refresh placeholder if needed
        try { if (typeof updateLengthPlaceholder === 'function') updateLengthPlaceholder(); } catch(e) {}
}

function showNewModelForm() {
    // If we were previously editing, clear that state so Create New always starts fresh
    try { editingModelId = null; originalTrainingData = null; } catch (e) {}
    try { clearNewModelForm(); } catch (e) {}

    const form = document.getElementById('new-model-form');
    if (!form) return;
    form.style.display = 'block';
    form.classList.add('active');
    const nameInput = document.getElementById('new-model-name');
    if (nameInput) nameInput.focus();
    // Ensure header shows Create mode
    try { setNewModelHeaderEditing(false); } catch (e) {}
    // Update the live names counter (in case the textarea was prefilled or needs resetting)
    try { if (typeof window.updateCustomNamesCount === 'function') window.updateCustomNamesCount(); } catch (e) {}

    // Smooth scroll the form into view and flash its outline once to draw attention
    try {
        form.scrollIntoView({ behavior: 'smooth', block: 'center' });
        // flash outline: set a temporary box-shadow and remove it after a short timeout
        const previousTransition = form.style.transition;
        form.style.transition = 'box-shadow 220ms ease-in-out';
        form.style.boxShadow = '0 0 0 4px rgba(255,200,0,0.95)';
        setTimeout(() => {
            form.style.boxShadow = '0 0 0 0 rgba(0,0,0,0)';
            // restore transition after a short delay
            setTimeout(() => { form.style.transition = previousTransition; }, 240);
        }, 300);
    } catch (e) {
        // not critical if scroll/flash fails in some browsers
        console.warn('Scroll/flash failed', e);
    }
}

function hideNewModelForm() {
    const form = document.getElementById('new-model-form');
    if (!form) return;
    form.style.display = 'none';
    form.classList.remove('active');
    clearNewModelForm();
}

function clearNewModelForm() {
    const name = document.getElementById('new-model-name'); if (name) name.value = '';
    const desc = document.getElementById('new-model-description'); if (desc) desc.value = '';
    const names = document.getElementById('custom-names-input'); if (names) names.value = '';
    const cat = document.getElementById('new-model-category'); if (cat) cat.value = '';

    // remove any error highlighting
    if (name) name.classList.remove('input-error');
    if (desc) desc.classList.remove('input-error');
    if (names) names.classList.remove('input-error');
    if (cat) cat.classList.remove('input-error');

    // Update counter after clearing
    try { if (typeof window.updateCustomNamesCount === 'function') window.updateCustomNamesCount(); } catch (e) {}
}

// Start training by POSTing to /train and streaming SSE updates
function startTraining(payload) {
    // Prevent multiple simultaneous trainings
    if (isTraining) {
        return;
    }

    isTraining = true;
    
    // Create new AbortController for this training request
    trainingAbortController = new AbortController();

    // Show loading UI (main.js defines these globals)
    if (typeof loadingDiv !== 'undefined' && loadingDiv) loadingDiv.style.display = 'block';
    if (typeof loadingText !== 'undefined' && loadingText) loadingText.textContent = 'Preparing training...';
    if (typeof progressBar !== 'undefined' && progressBar) progressBar.style.width = '0%';
    if (typeof restoreSpinner === 'function') restoreSpinner();
    
    // Show cancel training button
    const cancelTrainingBtn = document.getElementById('cancel-training-button');
    if (cancelTrainingBtn) cancelTrainingBtn.style.display = 'inline-block';
    
    // Hide generate cancel button (if visible for any reason)
    const cancelGenerateBtn = document.getElementById('cancel-generate-button');
    if (cancelGenerateBtn) cancelGenerateBtn.style.display = 'none';

    fetch('/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: trainingAbortController.signal
    }).then(response => {
        if (!response.ok) {
            throw new Error('Train request failed: ' + response.statusText);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        function processStream() {
            return reader.read().then(({ done, value }) => {
                if (done) return;
                const text = decoder.decode(value);
                const lines = text.split('\n\n');
                lines.forEach(line => {
                    if (!line.startsWith('data:')) return;
                    try {
                        const jsonData = JSON.parse(line.substring(5).trim());
                        if (jsonData.message && typeof loadingText !== 'undefined') {
                            loadingText.textContent = jsonData.message;
                        }
                        if (jsonData.progress !== undefined && typeof progressBar !== 'undefined') {
                            progressBar.style.width = jsonData.progress + '%';
                        }

                        switch (jsonData.type) {
                            case 'preparing':
                            case 'loading':
                            case 'training':
                                if (typeof restoreSpinner === 'function') restoreSpinner();
                                break;
                            case 'heartbeat':
                                // keep alive
                                break;
                            case 'error':
                                if (typeof showWarningImage === 'function') showWarningImage();
                                showStatusMessage('Training error: ' + (jsonData.message || 'Unknown'), 'error');
                                // Reset training state
                                resetTrainingState();
                                // Hide loading after showing error
                                setTimeout(() => {
                                    if (typeof loadingDiv !== 'undefined' && loadingDiv) {
                                        loadingDiv.style.display = 'none';
                                    }
                                }, 2000);
                                break;
                            case 'complete':
                                // training finished; backend should include model_id
                                const modelId = jsonData.model_id;
                                // Indicate training success and that artifacts were uploaded
                                showStatusMessage('Training complete — model saved', 'success');
                                // Refresh list and select model if id available
                                if (modelId) {
                                    loadCustomModels();
                                    selectModel(modelId);
                                    const hidden = document.getElementById('selected-custom-model');
                                    if (hidden) hidden.value = modelId;
                                } else {
                                    loadCustomModels();
                                }
                                // hide new model form
                                hideNewModelForm();
                                // Reset training state
                                resetTrainingState();
                                // hide loading after short pause to let user read
                                setTimeout(() => {
                                    if (typeof loadingDiv !== 'undefined' && loadingDiv) loadingDiv.style.display = 'none';
                                }, 800);
                                break;
                        }
                    } catch (e) {
                        console.error('Error parsing training SSE:', e);
                        if (typeof showWarningImage === 'function') showWarningImage();
                        showStatusMessage('Training stream parsing failed', 'error');
                    }
                });
                return processStream();
            });
        }

        return processStream();
    }).catch(err => {
        console.error('Training request failed:', err);
        
        // Reset training state
        resetTrainingState();
        
        // Check if error was due to abort
        if (err.name === 'AbortError') {
            // User cancelled, don't show error message
            return;
        }
        
        if (typeof showWarningImage === 'function') showWarningImage();
        showStatusMessage('Training failed: ' + err.message, 'error');
        
        // Hide loading bar after showing error
        setTimeout(() => {
            if (typeof loadingDiv !== 'undefined' && loadingDiv) {
                loadingDiv.style.display = 'none';
            }
        }, 2000);
    });
}

function saveCustomModel() {
    const nameEl = document.getElementById('new-model-name');
    const descEl = document.getElementById('new-model-description');
    const catEl = document.getElementById('new-model-category');
    const namesEl = document.getElementById('custom-names-input');
    if (!nameEl || !namesEl) return;

    const name = nameEl.value.trim();
    const description = descEl ? descEl.value.trim() : '';
    const category = catEl ? catEl.value : 'fantasy';
    const names = namesEl.value.trim();

    // Validation
    // clear previous error markers
    [nameEl, descEl, catEl, namesEl].forEach(el => { if (el) el.classList.remove('input-error'); });

    const errors = {};
    if (!name) {
        errors.name = 'Please enter a model name';
    }

    if (!names) {
        errors.trainingData = 'Please enter some training names';
    }

    const nameList = names.split('\n').filter(n => n.trim()).length;
    if (nameList < 3) {
        errors.tooFew = 'Please enter at least 3 names for training';
    }

    if (Object.keys(errors).length > 0) {
        // Highlight the offending fields
        if (errors.name && nameEl) nameEl.classList.add('input-error');
        if ((errors.trainingData || errors.tooFew) && namesEl) namesEl.classList.add('input-error');
        // Focus the first problematic field
        if (errors.name && nameEl) {
            nameEl.focus();
        } else if ((errors.trainingData || errors.tooFew) && namesEl) {
            namesEl.focus();
        }
        // Show an aggregated message
        const firstMsg = errors.name || errors.trainingData || errors.tooFew;
        showStatusMessage(firstMsg, 'error');
        return;
    }

    // Create new model
    // If we're editing an existing model, handle update vs. retrain
    if (editingModelId) {
        const existingId = editingModelId;
        // If training data is unchanged, only update metadata via PUT
        if (originalTrainingData !== null && originalTrainingData === names) {
            fetch('/api/custom_models/' + encodeURIComponent(existingId), {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name, description: description, category: category })
            })
            .then(r => r.json())
            .then(resp => {
                if (resp && resp.success) {
                    showStatusMessage('Model information updated', 'success');
                    loadCustomModels();
                    selectModel(existingId);
                } else {
                    showStatusMessage('Failed to update model information', 'error');
                }
            }).catch(err => {
                console.error('Update error:', err);
                showStatusMessage('Failed to update model: ' + err.message, 'error');
            }).finally(() => {
                editingModelId = null;
                originalTrainingData = null;
                hideNewModelForm();
            });
        } else {
            // training data changed: create/train as new model (existing behavior)
            // reset editing flags and proceed with create+train
            editingModelId = null;
            originalTrainingData = null;
            // reuse existing creation flow by posting to /api/custom_models
            const newModel = {
                id: generateModelId(),
                name: name,
                description: description,
                category: category,
                nameCount: nameList,
                createdAt: Date.now(),
                lastUsed: Date.now(),
                trainingData: names
            };

            fetch('/api/custom_models', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: newModel.name, description: newModel.description, category: newModel.category, trainingData: newModel.trainingData })
            }).then(r => r.json()).then(resp => {
                if (resp && resp.success) {
                    if (resp.model_id) newModel.id = resp.model_id;
                    loadCustomModels();
                    selectModel(newModel.id);
                    showStatusMessage('Model information saved — starting training...', 'info');
                    startTraining({ trainingData: newModel.trainingData, name: newModel.name, category: newModel.category, description: newModel.description });
                    // If we were editing an existing model, delete the old model artifacts to avoid duplicates
                    if (existingId && resp.model_id && resp.model_id !== existingId) {
                        fetch('/api/custom_models/' + encodeURIComponent(existingId), { method: 'DELETE' })
                            .then(r => r.json())
                            .then(delResp => {
                                if (delResp && delResp.success) {
                                    // remove from local list and refresh
                                    customModels = customModels.filter(m => m.id !== existingId);
                                    renderModelsList();
                                } else {
                                    console.warn('Failed to delete previous model:', delResp);
                                }
                            }).catch(err => {
                                console.error('Error deleting previous model:', err);
                            });
                    }
                } else {
                    showStatusMessage('Failed to save model: ' + (resp.error || 'unknown'), 'error');
                }
            }).catch(err => {
                console.error('Error saving model:', err);
                showStatusMessage('Failed to save model: ' + err.message, 'error');
            }).finally(() => {
                hideNewModelForm();
            });
        }
        return;
    }

    // Not editing: original create+train flow
    const newModel = {
        id: generateModelId(),
        name: name,
        description: description,
        category: category,
        nameCount: nameList,
        createdAt: Date.now(),
        lastUsed: Date.now(),
        trainingData: names // In production, this would be processed by backend
    };

    // Add to models array
    customModels.push(newModel);

    // Send metadata to backend and immediately start training
    fetch('/api/custom_models', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            name: newModel.name,
            description: newModel.description,
            category: newModel.category,
            trainingData: newModel.trainingData
        })
    })
    .then(r => r.json())
    .then(resp => {
        if (resp && resp.success) {
            // Do NOT show final "saved" success here — we only show it after .keras is uploaded
            // backend may return a deterministic model_id
            if (resp.model_id) newModel.id = resp.model_id;
            // refresh list so the UI knows about the backend model
            loadCustomModels();
            // auto-select the new model in the UI
            selectModel(newModel.id);
            // Inform the user that training will start
            showStatusMessage('Model metadata saved — starting training...', 'info');
            // Start training immediately and stream progress
            startTraining({
                trainingData: newModel.trainingData,
                name: newModel.name,
                category: newModel.category,
                description: newModel.description
            });
        } else {
            // If backend returns field-specific errors, highlight them
            if (resp && resp.errors && typeof resp.errors === 'object') {
                if (resp.errors.name && nameEl) nameEl.classList.add('input-error');
                if ((resp.errors.trainingData || resp.errors.tooFew) && namesEl) namesEl.classList.add('input-error');
                // Focus first server-reported field error
                if (resp.errors.name && nameEl) nameEl.focus();
                else if ((resp.errors.trainingData || resp.errors.tooFew) && namesEl) namesEl.focus();
                showStatusMessage(resp.error || 'Failed to save model due to validation errors', 'error');
            } else {
                showStatusMessage('Failed to save model: ' + (resp.error || 'unknown'), 'error');
            }
        }
    })
    .catch(err => {
        console.error('Error saving model:', err);
        showStatusMessage('Failed to save model: ' + err.message, 'error');
    });

    // Update UI
    renderModelsList();
    hideNewModelForm();

    // Auto-select the new model
    selectModel(newModel.id);
}

function editModel(modelId) {
    // Open the new-model form and populate with existing metadata/training data
    const model = customModels.find(m => m.id === modelId);
    // Show the edit form immediately with lightweight metadata, then fetch full metadata (including trainingData)
    const form = document.getElementById('new-model-form');
    if (!form) return;
    showNewModelForm();
    // Set header to editing immediately to avoid jank while metadata loads
    try { setNewModelHeaderEditing(true, modelId); } catch(e) {}
    const nameEl = document.getElementById('new-model-name');
    const descEl = document.getElementById('new-model-description');
    const catEl = document.getElementById('new-model-category');
    const namesEl = document.getElementById('custom-names-input');

    // Populate from lightweight in-memory metadata if available
    if (model) {
        if (nameEl) nameEl.value = model.name || '';
        if (descEl) descEl.value = model.description || '';
        if (catEl) catEl.value = model.category || '';
    }

    // Fetch the full metadata (server returns meta including trainingData)
    fetch('/api/custom_models/' + encodeURIComponent(modelId))
        .then(r => r.json())
        .then(resp => {
            if (resp && resp.meta) {
                const meta = resp.meta;
                if (nameEl) nameEl.value = meta.name || '';
                if (descEl) descEl.value = meta.description || '';
                if (catEl) catEl.value = meta.category || '';
                if (namesEl) namesEl.value = meta.trainingData || '';

                editingModelId = modelId;
                originalTrainingData = meta.trainingData || '';
                // Set header to editing mode and update count
                try { setNewModelHeaderEditing(true, modelId); } catch(e){}
                try { if (typeof window.updateCustomNamesCount === 'function') window.updateCustomNamesCount(); } catch(e){}
                if (nameEl) nameEl.focus();
            } else {
                showStatusMessage('Could not load model metadata for edit', 'error');
            }
        })
        .catch(err => {
            console.error('Failed to fetch metadata for edit:', err);
            showStatusMessage('Failed to load model metadata', 'error');
        });
}

function deleteModel(modelId) {
    if (!confirm('Are you sure you want to delete this model? This action cannot be undone.')) return;

    fetch('/api/custom_models/' + encodeURIComponent(modelId), { method: 'DELETE' })
        .then(r => r.json())
        .then(resp => {
            if (resp && resp.success) {
                showStatusMessage('Model deleted successfully', 'success');
                if (selectedModelId === modelId) {
            selectedModelId = null;
            // Clear the displayed model name but keep the info box visible so layout stays stable.
            const display = document.getElementById('selected-model-display'); if (display) display.textContent = '';
            const hidden = document.getElementById('selected-custom-model'); if (hidden) hidden.value = '';
            // Remove any concrete model hidden input so we don't accidentally submit a stale id
            const modelHidden = document.getElementById('model-hidden'); if (modelHidden) modelHidden.remove();
            // Do not change inline styles on #selected-model-info.
                }
                    // Ask the main UI to refresh avg-length placeholder when a model is removed
                    try { if (typeof updateLengthPlaceholder === 'function') updateLengthPlaceholder(); } catch(e) {}
                loadCustomModels();
            } else {
                showStatusMessage('Failed to delete model', 'error');
            }
        })
        .catch(err => {
            console.error('Delete error:', err);
            showStatusMessage('Failed to delete model: ' + err.message, 'error');
        });
}

function generateModelId() {
    return 'custom_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

function simulateSaveToBackend(model) {
    // POST to backend to save metadata and training data (S3)
    fetch('/api/custom_models', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            name: model.name,
            description: model.description,
            category: model.category,
            trainingData: model.trainingData
        })
    })
    .then(r => r.json())
    .then(resp => {
        if (resp && resp.success) {
            // update ID in local list if backend returned different id
            if (resp.model_id && resp.model_id !== model.id) {
                model.id = resp.model_id;
            }
            loadCustomModels();
        } else {
            showStatusMessage('Failed to save model: ' + (resp.error || 'unknown'), 'error');
        }
    })
    .catch(err => {
        console.error('Error saving model:', err);
        showStatusMessage('Failed to save model: ' + err.message, 'error');
    });
}

function showStatusMessage(message, type) {
    const statusContainer = document.getElementById('status-messages');
    if (!statusContainer) return;
    const messageId = 'msg_' + Date.now();

    const alertClass = type === 'success' ? 'success-message' : 
                     type === 'error' ? 'error-message-inline' : 
                     'custom-notice';

    const icon = type === 'success' ? 'check-circle' : 
                type === 'error' ? 'exclamation-triangle' : 
                'info-circle';

    const messageHtml = `
        <div id="${messageId}" class="${alertClass}">
            <i class="bi bi-${icon}"></i>
            ${message}
        </div>
    `;

    statusContainer.innerHTML = messageHtml;

    // // Auto-remove after 20 seconds
    // setTimeout(() => {
    //     const msgElement = document.getElementById(messageId);
    //     if (msgElement) msgElement.remove();
    // }, 20000);
}
