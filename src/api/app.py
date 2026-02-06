from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import json
import os

app = FastAPI()
MODEL_PATH = os.path.join("models", "fusion_model.pt")
LABELS_PATH = os.path.join("models", "labels.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
labels = None

class Features(BaseModel):
    features: list[float]  # flattened feature vector expected by the fusion model

@app.on_event("startup")
def load_model():
    global model, labels
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    try:
        # prefer scripted model but fall back to state dict
        try:
            model = torch.jit.load(MODEL_PATH, map_location=device)
        except Exception:
            model = torch.load(MODEL_PATH, map_location=device)
            model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed loading model: {e}")
    if os.path.exists(LABELS_PATH):
        try:
            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                labels = json.load(f)
        except Exception:
            labels = None

@app.get("/health")
async def health():
    return {"status": "ok", "device": str(device)}

@app.post("/predict")
async def predict(payload: Features):
    global model, labels
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    feat = np.asarray(payload.features, dtype=np.float32)
    if feat.ndim == 1:
        feat = feat[np.newaxis, :]
    tensor = torch.from_numpy(feat).to(device)
    try:
        model.eval()
        with torch.no_grad():
            out = model(tensor)
            # handle common output shapes
            if isinstance(out, (list, tuple)):
                out = out[0]
            out = out.cpu()
            if out.dim() == 1:
                out = out.unsqueeze(0)
            # if single-value regression
            if out.size(1) == 1:
                preds = out.squeeze(1).tolist()
                return {"predictions": preds}
            probs = torch.softmax(out, dim=1).numpy().tolist()
            top_idx = int(np.argmax(probs[0]))
            result = {"label_idx": top_idx, "probs": probs[0]}
            if labels and isinstance(labels, (list, dict)):
                try:
                    if isinstance(labels, list):
                        result["label"] = labels[top_idx]
                    elif isinstance(labels, dict):
                        result["label"] = labels.get(str(top_idx), None)
                except Exception:
                    pass
            return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")