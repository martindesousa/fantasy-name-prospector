// Model type toggle and tab switching
(function(){
    const toggleButtons = document.querySelectorAll('.toggle-option');
    const tabPanes = document.querySelectorAll('.tab-pane');
    const modelTypeInput = document.getElementById('model-type');
    const templateTab = document.getElementById('template-tab');
    const customTab = document.getElementById('custom-tab');

    // Remember last selected template model so we can restore when switching back
    let lastTemplateModel = (document.getElementById('model') && document.getElementById('model').value) || 'classic_american';

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
                lastTemplateModel = modelSelectEl ? modelSelectEl.value : lastTemplateModel;

                // If the select has an option with value 'custom', set it. Otherwise remove the select's name
                // but DO NOT create a placeholder hidden input with value 'custom'. We only want a concrete
                // hidden `model` field when the user explicitly selects a custom model (custom_models.js will
                // create that). Creating a placeholder caused the backend to receive model=custom.
                const hasCustomOption = modelSelectEl ? Array.from(modelSelectEl.options).some(o => o.value === 'custom') : false;
                if (hasCustomOption && modelSelectEl) {
                    modelSelectEl.value = 'custom';
                } else if (modelSelectEl) {
                    // remove name from select so it doesn't submit; do not create a placeholder hidden input
                    modelSelectEl.removeAttribute('name');
                    const existingHidden = document.getElementById('model-hidden');
                    if (existingHidden) existingHidden.remove();
                }

                const customNamesContainer = document.getElementById('custom-names-input') || document.getElementById('custom-names-container');
                const customNotice = document.getElementById('custom-notice');
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
                // Switching to template tab - clear any custom model status messages
                const statusContainer = document.getElementById('status-messages');
                if (statusContainer) {
                    statusContainer.innerHTML = '';
                }
                
                // restore template model
                const hidden = document.getElementById('model-hidden');
                if (hidden) {
                    hidden.remove();
                    const modelSelectEl2 = document.getElementById('model');
                    if (modelSelectEl2) modelSelectEl2.name = 'model';
                }
                const modelSelectEl3 = document.getElementById('model');
                if (modelSelectEl3) modelSelectEl3.value = lastTemplateModel;
                if (modelSelectEl3) modelSelectEl3.dispatchEvent(new Event('change', { bubbles: true }));
                if (modelSelectEl3 && modelSelectEl3.value !== 'custom') {
                    const customNamesContainer2 = document.getElementById('custom-names-input') || document.getElementById('custom-names-container');
                    const customNotice2 = document.getElementById('custom-notice');
                    if (customNamesContainer2) customNamesContainer2.style.display = 'none';
                    if (customNotice2) customNotice2.style.display = 'none';
                }
                const generateButton = document.getElementById('generate-button');
                if (generateButton) {
                    generateButton.style.display = 'inline-block';
                    generateButton.disabled = false;
                }
                // Update model-dependent UI state
                try { if (typeof updateLengthPlaceholder === 'function') updateLengthPlaceholder(); } catch (e) {}
            }
        });
    });
})();
