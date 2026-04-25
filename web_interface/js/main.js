async function refresh() {
    setStatus("loading");
    try {
        const [candle, predictions, ...patternResults] = await Promise.all([
            fetchCandle(),
            fetchPredictions(),
            ...PATTERN_CONFIGS.map(cfg => fetchPattern(cfg.endpoint))
        ]);
        updateCandle(candle);
        updateModels(predictions);
        updatePatternCards(patternResults);
        setLastUpdated(candle.timestamp);
        setStatus("ok");
    } catch (err) {
        setStatus("error");
        console.error("Refresh failed:", err);
    }
}

document.addEventListener("DOMContentLoaded", () => {
    buildModelCards();
    buildPatternCards();
    refresh();
    setInterval(refresh, REFRESH_INTERVAL_MS);
    document.getElementById("refresh-btn").addEventListener("click", refresh);
});
