from pydantic import BaseModel

class PredictionResponse(BaseModel):
    xgbGatePred: str # "FLAT" | "DIR"
    xgbGateProbs: dict[str, float] # {"0": 0.26, "1": 0.74}
    xgbGateVersion: str
    patchTstGatePred: str # "FLAT" | "DIR"
    patchTstGateProbs: dict[str, float] # {"0": 0.26, "1": 0.74}
    patchTstGateVersion: str
    timestamp: str

class CandleInfo(BaseModel):
    open: float
    high: float
    low: float
    close: float
    barrier: float
    timestamp: str
