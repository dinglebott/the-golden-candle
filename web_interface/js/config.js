const API_BASE_URL = "https://your-api-url.com";

const REFRESH_INTERVAL_MS = 60 * 1000;

// To add a new model: add an entry to MODEL_CONFIGS.
// predKey, probsKey, versionKey must match the keys returned by /predict.
// classes maps each class index string to a display label and colour.
const MODEL_CONFIGS = [
    {
        id: "xgbGate",
        label: "XGBoost Gate",
        versionKey: "xgbGateVersion",
        predKey: "xgbGatePred",
        probsKey: "xgbGateProbs",
        classes: {
            "0": { label: "FLAT", color: "#94a3b8" },
            "1": { label: "DIR",  color: "#4ade80" }
        }
    }
];
