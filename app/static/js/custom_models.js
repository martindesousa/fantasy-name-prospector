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

function loadCustomModels() {
    // In production, this would load from your backend
    // For now, using mock data with simulated persistence
    renderModelsList();
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
            <div class="custom-model-item ${selectedModelId === model.id ? 'selected' : ''}" 
                 data-model-id="${model.id}">
                <div class="model-info">
                    <div class="model-name">${model.name}</div>
                    <div class="model-meta">
                        ${model.description || 'No description'}
                    </div>
                    <div class="model-stats">
                        <span class="stat-badge">${model.nameCount} names</span>
                        <span class="stat-badge">${model.category}</span>
                        <span class="stat-badge">Used ${lastUsedDate}</span>
                    </div>
                </div>
                <div class="model-actions">
                    <button class="btn btn-outline-primary btn-icon" data-action="select" data-id="${model.id}" title="Select">
                        <i class="bi bi-check2"></i>
                    </button>
                    <button class="btn btn-outline-secondary btn-icon" data-action="edit" data-id="${model.id}" title="Edit">
                        <i class="bi bi-pencil"></i>
                    </button>
                    <button class="btn btn-outline-danger btn-icon" data-action="delete" data-id="${model.id}" title="Delete">
                        <i class="bi bi-trash"></i>
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

    if (model) {
        // Update UI
        const display = document.getElementById('selected-model-display');
        if (display) display.textContent = model.name;
        const info = document.getElementById('selected-model-info');
        if (info) info.style.display = 'block';
        const hidden = document.getElementById('selected-custom-model');
        if (hidden) hidden.value = modelId;

        // Update last used
        model.lastUsed = Date.now();

        // Re-render to update selection
        renderModelsList();

        // Hide new model form if open
        hideNewModelForm();
    }
}

function showNewModelForm() {
    const form = document.getElementById('new-model-form');
    if (!form) return;
    form.style.display = 'block';
    form.classList.add('active');
    const nameInput = document.getElementById('new-model-name');
    if (nameInput) nameInput.focus();
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
    const cat = document.getElementById('new-model-category'); if (cat) cat.value = 'fantasy';
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
    if (!name) {
        showStatusMessage('Please enter a model name', 'error');
        return;
    }

    if (!names) {
        showStatusMessage('Please enter some training names', 'error');
        return;
    }

    const nameList = names.split('\n').filter(n => n.trim()).length;
    if (nameList < 5) {
        showStatusMessage('Please enter at least 5 names for training', 'error');
        return;
    }

    // Create new model
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

    // In production, send to backend here
    simulateSaveToBackend(newModel);

    // Update UI
    renderModelsList();
    showStatusMessage(`Model "${name}" saved successfully!`, 'success');
    hideNewModelForm();

    // Auto-select the new model
    selectModel(newModel.id);
}

function editModel(modelId) {
    // In production, this would open an edit form
    showStatusMessage('Edit functionality coming soon!', 'info');
}

function deleteModel(modelId) {
    if (!confirm('Are you sure you want to delete this model? This action cannot be undone.')) return;
    customModels = customModels.filter(m => m.id !== modelId);

    // In production, send delete request to backend

    if (selectedModelId === modelId) {
        selectedModelId = null;
        const info = document.getElementById('selected-model-info'); if (info) info.style.display = 'none';
        const hidden = document.getElementById('selected-custom-model'); if (hidden) hidden.value = '';
    }

    renderModelsList();
    showStatusMessage('Model deleted successfully', 'success');
}

function generateModelId() {
    return 'custom_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

function simulateSaveToBackend(model) {
    // In production, this would be an actual API call
    console.log('Saving model to backend:', model);

    // Simulate network delay
    setTimeout(() => {
        console.log('Model saved to backend successfully');
    }, 1000);
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

    // Auto-remove after 5 seconds
    setTimeout(() => {
        const msgElement = document.getElementById(messageId);
        if (msgElement) msgElement.remove();
    }, 5000);
}
