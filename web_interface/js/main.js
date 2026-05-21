async function refresh() {
    setStatus("loading");
    try {
        const [candle, predictions, ...cardResults] = await Promise.all([
            fetchCandle(),
            fetchPredictions(),
            ...PATTERN_CONFIGS.map(cfg => fetchPattern(cfg.endpoint)),
            ...STRATEGY_CONFIGS.map(cfg => fetchPattern(cfg.endpoint))
        ]);
        const patternResults  = cardResults.slice(0, PATTERN_CONFIGS.length);
        const strategyResults = cardResults.slice(PATTERN_CONFIGS.length);
        updateCandle(candle);
        updateModels(predictions);
        updatePatternCards(patternResults);
        updateStrategyCards(strategyResults);
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
    buildStrategyCards();
    refresh();
    setInterval(refresh, REFRESH_INTERVAL_MS);
    document.getElementById("refresh-btn").addEventListener("click", refresh);
});
