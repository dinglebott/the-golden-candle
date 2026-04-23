async function refresh() {
    setStatus("loading");
    try {
        const [candle, predictions] = await Promise.all([fetchCandle(), fetchPredictions()]);
        updateCandle(candle);
        updateModels(predictions);
        setLastUpdated(candle.timestamp);
        setStatus("ok");
    } catch (err) {
        setStatus("error");
        console.error("Refresh failed:", err);
    }
}

document.addEventListener("DOMContentLoaded", () => {
    buildModelCards();
    refresh();
    setInterval(refresh, REFRESH_INTERVAL_MS);
    document.getElementById("refresh-btn").addEventListener("click", refresh);
});
