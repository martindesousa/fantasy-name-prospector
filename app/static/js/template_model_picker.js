(function () {
    const select = document.getElementById('model');
    const selectedPanel = document.getElementById('template-selected-panel');
    const selectedName = document.getElementById('template-selected-name');
    const selectedMeta = document.getElementById('template-selected-meta');
    const browser = document.getElementById('template-browser');
    const openButton = document.getElementById('open-template-browser');
    const searchInput = document.getElementById('template-model-search');
    const clearButton = document.getElementById('clear-template-search');
    const status = document.getElementById('template-browser-status');
    const results = document.getElementById('template-browser-results');
    const emptyState = document.getElementById('template-browser-empty');

    if (!select || !selectedPanel || !selectedName || !selectedMeta || !browser || !openButton || !searchInput || !clearButton || !status || !results || !emptyState) {
        return;
    }

    const models = Array.from(select.options)
        .filter((option) => option.value)
        .map((option) => ({
            id: option.value,
            name: option.textContent.trim(),
            type: option.dataset.type || 'Other',
            group: option.dataset.group || 'Other',
            description: option.dataset.description || 'No description available.',
            searchText: [
                option.textContent,
                option.dataset.type,
                option.dataset.group,
                option.dataset.description
            ].join(' ').toLowerCase()
        }));

    let searchTerm = '';

    function getSelectedModel() {
        return models.find((model) => model.id === select.value) || models[0] || null;
    }

    function setBrowserVisibility(isVisible) {
        browser.hidden = !isVisible;
        openButton.textContent = isVisible ? 'Hide Browser' : 'Browse Templates';

        if (isVisible) {
            searchInput.focus();
        }
    }

    function updateSelectedSummary() {
        const selectedModel = getSelectedModel();
        if (!selectedModel) {
            return;
        }

        selectedName.textContent = selectedModel.name;
        selectedMeta.textContent = `${selectedModel.type} | ${selectedModel.group}`;
    }

    function getVisibleModels() {
        const normalized = searchTerm.trim().toLowerCase();
        return models.filter((model) => !normalized || model.searchText.includes(normalized));
    }

    function groupByTypeAndGroup(modelsToRender) {
        const typeMap = new Map();

        modelsToRender.forEach((model) => {
            if (!typeMap.has(model.type)) {
                typeMap.set(model.type, new Map());
            }

            const groupMap = typeMap.get(model.type);
            if (!groupMap.has(model.group)) {
                groupMap.set(model.group, []);
            }

            groupMap.get(model.group).push(model);
        });

        return Array.from(typeMap.entries())
            .sort((left, right) => left[0].localeCompare(right[0]))
            .map(([type, groups]) => ({
                type,
                groups: Array.from(groups.entries())
                    .sort((left, right) => left[0].localeCompare(right[0]))
                    .map(([group, items]) => ({
                        group,
                        items: items.sort((left, right) => left.name.localeCompare(right.name))
                    }))
            }));
    }

    function buildModelCard(model) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'template-model-card';
        button.dataset.modelId = model.id;

        if (select.value === model.id) {
            button.classList.add('selected');
        }

        const name = document.createElement('div');
        name.className = 'template-model-card-name';
        name.textContent = model.name;

        const description = document.createElement('p');
        description.className = 'template-model-card-description';
        description.textContent = model.description;

        button.appendChild(name);
        button.appendChild(description);

        return button;
    }

    function renderBrowser() {
        const visibleModels = getVisibleModels();
        const grouped = groupByTypeAndGroup(visibleModels);
        const normalized = searchTerm.trim();

        results.innerHTML = '';

        if (!visibleModels.length) {
            emptyState.hidden = false;
            results.hidden = true;
            status.textContent = normalized ? `0 models match "${normalized}"` : '0 models available';
            return;
        }

        results.hidden = false;
        emptyState.hidden = true;
        status.textContent = normalized ? `${visibleModels.length} models match "${normalized}"` : `${visibleModels.length} models available`;

        grouped.forEach((typeBlock) => {
            const typeSection = document.createElement('section');
            typeSection.className = 'template-type-section';

            const typeHeading = document.createElement('div');
            typeHeading.className = 'template-type-heading';
            typeHeading.textContent = typeBlock.type;
            typeSection.appendChild(typeHeading);

            typeBlock.groups.forEach((groupBlock) => {
                const groupSection = document.createElement('section');
                groupSection.className = 'template-group-section';

                const groupHeading = document.createElement('div');
                groupHeading.className = 'template-group-heading';
                groupHeading.textContent = groupBlock.group;

                const groupGrid = document.createElement('div');
                groupGrid.className = 'template-model-grid';

                groupBlock.items.forEach((model) => {
                    groupGrid.appendChild(buildModelCard(model));
                });

                groupSection.appendChild(groupHeading);
                groupSection.appendChild(groupGrid);
                typeSection.appendChild(groupSection);
            });

            results.appendChild(typeSection);
        });
    }

    function selectModel(modelId) {
        const model = models.find((entry) => entry.id === modelId);
        if (!model) {
            return;
        }

        select.value = model.id;
        updateSelectedSummary();
        renderBrowser();
        setBrowserVisibility(false);
        select.dispatchEvent(new Event('change', { bubbles: true }));
    }

    openButton.addEventListener('click', function () {
        setBrowserVisibility(browser.hidden);
    });

    searchInput.addEventListener('input', function () {
        searchTerm = this.value;
        clearButton.hidden = !searchTerm.trim();
        renderBrowser();
    });

    searchInput.addEventListener('keydown', function (event) {
        if (event.key === 'Escape' && this.value) {
            this.value = '';
            searchTerm = '';
            clearButton.hidden = true;
            renderBrowser();
        }

        if (event.key === 'Enter') {
            event.preventDefault();
            const firstMatch = getVisibleModels()[0];
            if (firstMatch) {
                selectModel(firstMatch.id);
            }
        }
    });

    clearButton.addEventListener('click', function () {
        searchInput.value = '';
        searchTerm = '';
        clearButton.hidden = true;
        renderBrowser();
        searchInput.focus();
    });

    results.addEventListener('click', function (event) {
        const card = event.target.closest('[data-model-id]');
        if (!card) {
            return;
        }

        selectModel(card.dataset.modelId);
    });

    select.addEventListener('change', function () {
        updateSelectedSummary();
        renderBrowser();
        if (getSelectedModel()) {
            setBrowserVisibility(false);
        }
    });

    updateSelectedSummary();
    renderBrowser();
    setBrowserVisibility(!getSelectedModel());
})();