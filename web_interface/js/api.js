async function fetchCandle() {
    const res = await fetch(`${API_BASE_URL}/candle`);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json();
}

async function fetchPredictions() {
    const res = await fetch(`${API_BASE_URL}/predict`);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json();
}
