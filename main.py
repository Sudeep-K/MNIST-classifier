from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import joblib
import io
import numpy as np

app = FastAPI()
model = joblib.load("models/mnist_model.pkl")


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
async def predict(file: UploadFile):
    bytes_data = await file.read()
    input_data = preprocess_image(bytes_data)
    probs = model.predict_proba(input_data)[0]
    prediction = model.predict(input_data)[0]

    return JSONResponse(
        {
            "prediction": int(prediction),
            "probabilities": {str(i): float(p) for i, p in enumerate(probs)},
        }
    )
