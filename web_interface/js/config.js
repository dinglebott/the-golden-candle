const API_BASE_URL = "https://the-golden-candle.up.railway.app";

const REFRESH_INTERVAL_MS = 30 * 60 * 1000; // every 30min

// Models that always produce a prediction (no no-signal state).
// predKey, probsKey, versionKey must match keys returned by /predict.
const GATE_CONFIGS = [
    {
        id: "patchTstGate",
        label: "PatchTST Gate",
        versionKey: "patchTstGateVersion",
        predKey: "patchTstGatePred",
        probsKey: "patchTstGateProbs",
        classes: {
            "0": { label: "FLAT", color: "#94a3b8" },
            "1": { label: "DIR",  color: "#4ade80" }
        }
    }
];

// Pattern detectors: conditional on pattern detection.
// API response shape (PatternResponse): { detected, pred, probs, version, meta, timestamp }
//
// To add a new pattern:
//   1. Implement backend detector + register in PATTERN_REGISTRY (main.py) and PATTERN_VERSIONS (inference.py).
//   2. Add an entry here:
//      - endpoint: matches /pattern/<name> on the server
//      - classes: maps probs keys ("0", "1") to display label and bar color
//      - renderMeta(metaEl, meta): optional; renders pattern-specific metadata into the card footer.
//        Called only when detected=true. meta is the object from the API response.
const PATTERN_CONFIGS = [
    {
        id: "fvg",
        label: "CNN-LSTM FVG",
        endpoint: "/pattern/fvg",
        classes: {
            "0": { label: "NO_FILL", badge: "X", color: "#f87171" },
            "1": { label: "FILL", badge: "FILL", color: "#4ade80" },
        },
        renderMeta(metaEl, meta) {
            const dirColor = meta.direction === "bullish" ? "#4ade80" : "#f87171";
            // detection_time is the open of candle i; +1h gives the start of the fill window
            const dt = new Date(meta.detection_time);
            dt.setUTCHours(dt.getUTCHours() + 1);
            const timeStr = dt.toLocaleString("en-SG", {
                hour: "2-digit", minute: "2-digit", hour12: false,
                timeZone: "Asia/Singapore"
            });
            metaEl.innerHTML = `
                <span class="pattern-direction" style="color: ${dirColor}">${meta.direction.toUpperCase()}</span>
                <span class="pattern-gap">${formatPrice(meta.gap_low)} – ${formatPrice(meta.gap_high)}</span>
                <span class="pattern-time">${timeStr} SGT</span>
            `;
        }
    },
];
