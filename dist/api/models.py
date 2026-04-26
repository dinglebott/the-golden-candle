from pydantic import BaseModel

class PredictionResponse(BaseModel):
    patchTstGatePred: str # "FLAT" | "DIR"
    patchTstGateProbs: dict[str, float] # {"0": 0.26, "1": 0.74}
    patchTstGateVersion: str
    timestamp: str

class PatternResponse(BaseModel):
    detected: bool
    pred: str # Pattern-specific label (e.g. "FILL" / "NO_FILL") or "NO_SIGNAL"
    probs: dict[str, float] | None # {"0": p_neg, "1": p_pos}; null when not detected
    version: str
    meta: dict | None # Pattern-specific metadata (e.g. gap bounds, direction); null when not detected
    timestamp: str

class CandleInfo(BaseModel):
    open: float
    high: float
    low: float
    close: float
    barrier: float
    timestamp: str
