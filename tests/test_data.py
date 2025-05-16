import sys
import os

# Add the parent directory to the path so Python can find main.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_prediction_accuracy():
    response = client.get("/prediction-accuracy/")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions" in data
    assert "correct_predictions" in data
    assert "incorrect_predictions" in data
    assert "accuracy_percent" in data
    assert isinstance(data["total_predictions"], int)
    assert isinstance(data["accuracy_percent"], float)
