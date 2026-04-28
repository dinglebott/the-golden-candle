function formatPrice(val) {
    return val.toFixed(5);
}

function formatTimestamp(dateObj) {
    return dateObj.toLocaleString("en-SG", {
        day: "2-digit", month: "short",
        hour: "2-digit", minute: "2-digit", hour12: false
    });
}

// ── Gate model cards (always-on) ──────────────────────────────────────────────

function buildModelCards() {
    const container = document.getElementById("model-cards");
    container.innerHTML = "";

    GATE_CONFIGS.forEach(cfg => {
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

function updateModels(data) {
    GATE_CONFIGS.forEach(cfg => {
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

// ── Pattern detector cards (conditional on detection) ────────────────────────

function buildPatternCards() {
    const container = document.getElementById("model-cards");

    PATTERN_CONFIGS.forEach(cfg => {
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
            <div class="tp-sl-rows">
                <div class="tp-sl-item">
                    <span class="tp-sl-label">TP</span>
                    <span class="tp-sl-value tp" id="tp-${cfg.id}">--</span>
                </div>
                <div class="tp-sl-item">
                    <span class="tp-sl-label">SL</span>
                    <span class="tp-sl-value sl" id="sl-${cfg.id}">--</span>
                </div>
            </div>
            <div class="pattern-meta" id="meta-${cfg.id}" style="display: none"></div>
        `;
        container.appendChild(card);
    });
}

function updatePatternCard(cfg, data) {
    const predEl    = document.getElementById(`pred-${cfg.id}`);
    const versionEl = document.getElementById(`version-${cfg.id}`);
    const metaEl    = document.getElementById(`meta-${cfg.id}`);

    versionEl.textContent = data.version ? `v${data.version}` : "--";

    const tpEl = document.getElementById(`tp-${cfg.id}`);
    const slEl = document.getElementById(`sl-${cfg.id}`);

    if (!data.detected) {
        predEl.textContent       = "--";
        predEl.style.color       = "var(--text-muted)";
        predEl.style.borderColor = "var(--text-muted)";
        Object.keys(cfg.classes).forEach(k => {
            const bar = document.getElementById(`bar-${cfg.id}-${k}`);
            const val = document.getElementById(`prob-val-${cfg.id}-${k}`);
            if (bar) bar.style.width = "0%";
            if (val) val.textContent = "--";
        });
        if (tpEl) tpEl.textContent = "--";
        if (slEl) slEl.textContent = "--";
        metaEl.style.display = "none";
        return;
    }

    const matchedClass = Object.values(cfg.classes).find(cls => cls.label === data.pred);
    const predColor = matchedClass ? matchedClass.color : "var(--text-muted)";

    predEl.textContent       = matchedClass?.badge;
    predEl.style.color       = predColor;
    predEl.style.borderColor = predColor;

    Object.entries(data.probs).forEach(([k, p]) => {
        const pct = (p * 100).toFixed(1);
        const bar = document.getElementById(`bar-${cfg.id}-${k}`);
        const val = document.getElementById(`prob-val-${cfg.id}-${k}`);
        if (bar) bar.style.width = `${pct}%`;
        if (val) val.textContent = `${pct}%`;
    });

    if (tpEl && data.meta?.tp != null) tpEl.textContent = formatPrice(data.meta.tp);
    if (slEl && data.meta?.sl != null) slEl.textContent = formatPrice(data.meta.sl);

    if (cfg.renderMeta && data.meta) {
        cfg.renderMeta(metaEl, data.meta);
        metaEl.style.display = "flex";
    }
}

function updatePatternCards(results) {
    PATTERN_CONFIGS.forEach((cfg, i) => updatePatternCard(cfg, results[i]));
}

// ── Candle card ───────────────────────────────────────────────────────────────

function updateCandle(data) {
    document.getElementById("candle-open").textContent    = formatPrice(data.open);
    document.getElementById("candle-high").textContent    = formatPrice(data.high);
    document.getElementById("candle-low").textContent     = formatPrice(data.low);
    document.getElementById("candle-close").textContent   = formatPrice(data.close);
    document.getElementById("candle-barrier").textContent = formatPrice(data.barrier);

    const candleStart = new Date(data.timestamp);
    const candleEnd   = new Date(candleStart);
    candleEnd.setHours(candleStart.getHours() + 1);
    const startString = candleStart.toLocaleString("en-SG", {
        hour: "2-digit", minute: "2-digit", hour12: false
    });
    const endString = candleEnd.toLocaleString("en-SG", {
        hour: "2-digit", minute: "2-digit", hour12: false
    });
    document.getElementById("candle-timestamp").textContent = `${startString} - ${endString} SGT`;
}

// ── Status ────────────────────────────────────────────────────────────────────

function setStatus(state) {
    document.getElementById("status-dot").className = `status-dot ${state}`;
    document.getElementById("status-text").textContent =
        state === "loading" ? "Updating..." :
        state === "error"   ? "Update failed" : "";
}

function setLastUpdated(timestamp) {
    const currentTime = new Date(Date.now()).toLocaleString("en-SG", {
        day: "2-digit", month: "short",
        hour: "2-digit", minute: "2-digit", hour12: false
    });
    document.getElementById("last-updated").textContent = `Last updated: ${currentTime} SGT`;
}
