function formatPrice(val) {
    return val.toFixed(5);
}

function formatTimestamp(ts) {
    return new Date(ts).toLocaleString("en-GB", {
        day: "2-digit", month: "short", year: "numeric",
        hour: "2-digit", minute: "2-digit", timeZone: "UTC"
    }) + " UTC";
}

// Called once on load — builds a card DOM node for each model in MODEL_CONFIGS
function buildModelCards() {
    const container = document.getElementById("model-cards");
    container.innerHTML = "";

    MODEL_CONFIGS.forEach(cfg => {
        const probRows = Object.entries(cfg.classes).map(([k, cls]) => `
            <div class="prob-row">
                <span class="prob-label">${cls.label}</span>
                <div class="prob-track">
                    <div class="prob-fill" id="bar-${cfg.id}-${k}" style="background: ${cls.color}; width: 0%"></div>
                </div>
                <span class="prob-value" id="prob-val-${cfg.id}-${k}">--</span>
            </div>
        `).join("");

        const card = document.createElement("div");
        card.className = "card model-card";
        card.id = `model-card-${cfg.id}`;
        card.innerHTML = `
            <div class="card-header">
                <span class="card-title">${cfg.label}</span>
                <span class="version-badge" id="version-${cfg.id}">--</span>
            </div>
            <div class="model-body">
                <div class="prediction-badge" id="pred-${cfg.id}">--</div>
                <div class="prob-bars">${probRows}</div>
            </div>
        `;
        container.appendChild(card);
    });
}

function updateCandle(data) {
    document.getElementById("candle-open").textContent    = formatPrice(data.open);
    document.getElementById("candle-high").textContent    = formatPrice(data.high);
    document.getElementById("candle-low").textContent     = formatPrice(data.low);
    document.getElementById("candle-close").textContent   = formatPrice(data.close);
    document.getElementById("candle-barrier").textContent = formatPrice(data.barrier);
    document.getElementById("candle-timestamp").textContent = formatTimestamp(data.timestamp);
}

function updateModels(data) {
    MODEL_CONFIGS.forEach(cfg => {
        const pred    = data[cfg.predKey];
        const probs   = data[cfg.probsKey];
        const version = data[cfg.versionKey];

        const predEl    = document.getElementById(`pred-${cfg.id}`);
        const versionEl = document.getElementById(`version-${cfg.id}`);

        const matchedClass = Object.values(cfg.classes).find(cls => cls.label === pred);
        const predColor = matchedClass ? matchedClass.color : "var(--text-muted)";

        predEl.textContent = pred ?? "--";
        predEl.style.color = predColor;
        predEl.style.borderColor = predColor;
        versionEl.textContent = version ? `v${version}` : "--";

        Object.entries(probs).forEach(([k, p]) => {
            const pct = (p * 100).toFixed(1);
            const bar = document.getElementById(`bar-${cfg.id}-${k}`);
            const val = document.getElementById(`prob-val-${cfg.id}-${k}`);
            if (bar) bar.style.width = `${pct}%`;
            if (val) val.textContent = `${pct}%`;
        });
    });
}

function setStatus(state) {
    // state: "loading" | "ok" | "error"
    document.getElementById("status-dot").className = `status-dot ${state}`;
    document.getElementById("status-text").textContent =
        state === "loading" ? "Updating..." :
        state === "error"   ? "Update failed" : "";
}

function setLastUpdated(timestamp) {
    document.getElementById("last-updated").textContent = `Last updated: ${formatTimestamp(timestamp)}`;
}
