from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
import joblib
import io
import numpy as np
from db import init_db, Prediction, SessionLocal

init_db()

app = FastAPI()
model = joblib.load("models/mnist_model.pkl")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("L").resize((28, 28))
    image_array = np.array(image)
    image_array = 255 - image_array  # Invert colors if needed
    flat = image_array.flatten() / 255.0
    return flat.reshape(1, -1)


@app.get("/")
def hello():
    return {"hello": "world"}


@app.post("/predict/")
async def predict(file: UploadFile, true_label: int = Form(...)):
    bytes_data = await file.read()
    input_data = preprocess_image(bytes_data)
    probs = model.predict_proba(input_data)[0]
    prediction = model.predict(input_data)[0]

    db = SessionLocal()
    try:
        db_pred = Prediction(predicted_label=int(prediction), true_label=true_label)
        db.add(db_pred)
        db.commit()
    finally:
        db.close()

    return JSONResponse(
        {
            "prediction": int(prediction),
            "true_label": true_label,
            "probabilities": {str(i): float(p) for i, p in enumerate(probs)},
        }
    )


@app.get("/prediction-accuracy/")
def get_prediction_accuracy():
    db = SessionLocal()
    try:
        total = db.query(Prediction).count()
        correct = (
            db.query(Prediction)
            .filter(Prediction.predicted_label == Prediction.true_label)
            .count()
        )
        incorrect = total - correct
    finally:
        db.close()

    return {
        "total_predictions": total,
        "correct_predictions": correct,
        "incorrect_predictions": incorrect,
        "accuracy_percent": round((correct / total * 100), 2) if total > 0 else 0.0,
    }
