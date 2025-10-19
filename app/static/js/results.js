// Results renderer: creates and updates the generated names UI
(function(){
    function ensureContainer() {
        let resultsContainer = document.querySelector('.results-section');
        if (!resultsContainer) {
            resultsContainer = document.createElement('div');
            resultsContainer.className = 'main-container results-section';
            resultsContainer.innerHTML = `
                <div class="results-header">
                    <h3>Generated Names</h3>
                    <span class="badge bg-primary" id="name-count">0 names</span>
                </div>
                <div class="name-grid" id="generated-name-grid"></div>
            `;
            document.querySelector('.container-fluid').appendChild(resultsContainer);

            // spacer to soften scroll landing
            let spacer = document.getElementById('results-spacer');
            if (!spacer) {
                spacer = document.createElement('div');
                spacer.id = 'results-spacer';
                spacer.style.height = '240px';
                spacer.style.pointerEvents = 'none';
                document.body.appendChild(spacer);
            }

            try {
                resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
                const previousTransition = resultsContainer.style.transition;
                resultsContainer.style.transition = 'box-shadow 220ms ease-in-out';
                resultsContainer.style.boxShadow = '0 0 0 4px rgba(23, 109, 201, 1)';
                setTimeout(() => {
                    resultsContainer.style.boxShadow = '0 0 0 0 rgba(0,0,0,0)';
                    setTimeout(() => { resultsContainer.style.transition = previousTransition; }, 240);
                }, 360);
            } catch (e) {
                // ignore scroll failures
            }
        }
        return resultsContainer;
    }

    function addName(name) {
        const container = ensureContainer();
        const nameGrid = container.querySelector('#generated-name-grid');
        const nameItem = document.createElement('div');
        nameItem.className = 'name-item';
        nameItem.textContent = name;
        nameGrid.appendChild(nameItem);

        const countBadge = container.querySelector('#name-count');
        if (countBadge) {
            const currentCount = nameGrid.children.length;
            countBadge.textContent = `${currentCount} names`;
        }
    }

    function clear() {
        const existing = document.querySelector('.results-section');
        if (existing) existing.remove();
        const spacer = document.getElementById('results-spacer');
        if (spacer) spacer.remove();
    }

    function renderFinal(namesArray) {
        clear();
        const finalResultsContainer = document.createElement('div');
        finalResultsContainer.className = 'main-container results-section';
        finalResultsContainer.innerHTML = `
            <div class="results-header">
                <h3>Generated Names</h3>
                <span class="badge bg-primary">${namesArray.length} names</span>
            </div>
            <div class="name-grid">
                ${namesArray.map(name => `<div class="name-item" onclick="navigator.clipboard.writeText('${name.replace(/'/g, "\\'")}')">${name}</div>`).join('')}
            </div>
        `;
        document.querySelector('.container-fluid').appendChild(finalResultsContainer);
    }

    window.results = {
        addName,
        clear,
        renderFinal,
        ensureContainer
    };
})();
