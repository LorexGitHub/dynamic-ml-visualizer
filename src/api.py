from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import manager

app = FastAPI(title="Cancer Prediction API")

class TrainConfig(BaseModel):
    epochs: int = 50
    lr: float = 0.01
    hidden_dim: int = 32

class PredictFeatures(BaseModel):
    features: list[float]

@app.get("/info")
def get_info():
    return {"status": "loaded", "data": manager.load_data()}

@app.get("/plot/pca")
def get_pca():
    return {"image_base64": manager.get_pca_plot()}

@app.post("/train")
def train_model(config: TrainConfig):
    acc, cm = manager.train(epochs=config.epochs, lr=config.lr, hidden_dim=config.hidden_dim)
    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "training_plot_base64": manager.get_training_plot(),
        "confusion_plot_base64": manager.get_confusion_plot(cm)
    }

@app.post("/predict")
def predict(data: PredictFeatures):
    if len(data.features) != 30: raise HTTPException(400, "Exactly 30 features required")
    res = manager.predict(data.features)
    if not res: raise HTTPException(400, "Model not trained yet")
    return res