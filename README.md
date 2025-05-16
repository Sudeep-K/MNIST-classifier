# MNIST Digit Classifier API with FastAPI and PostgreSQL

This project is a **machine learning-powered web API** built with **FastAPI** that lets users upload handwritten digit images (MNIST style), get digit predictions from a trained model, and store the predicted and true labels in a **PostgreSQL** database. It also provides an endpoint to check prediction accuracy based on stored results.

---

## Features

- Upload image files for digit prediction.
- Input the true digit label for validation.
- Store predicted and true labels with timestamps in PostgreSQL.
- Retrieve overall prediction accuracy from saved data.
- Dockerized for easy local development with Docker Compose.
- Uses SQLAlchemy ORM and `psycopg2` for database interaction.
- Environment variables managed securely using `.env` and `python-dotenv`.

---

## Tech Stack

- Python 3.10
- FastAPI
- Uvicorn (ASGI server)
- scikit-learn (for ML model)
- PostgreSQL 15
- SQLAlchemy ORM
- psycopg2
- Docker & Docker Compose

---

## Getting Started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed
- [Docker Compose](https://docs.docker.com/compose/install/) installed
- Python 3.10+ (optional, if running without Docker)

---

## Setup Environment and Run with Docker Compose

### Step 1: Create `.env` file

Create a `.env` file in the project root with the following content:

```env
POSTGRES_USER=
POSTGRES_PASSWORD=
POSTGRES_DB=
POSTGRES_URL=
```

### Step 2: Build and run with Docker Compose

Run the following command in the project root to build the Docker images and start the containers:

```bash
docker-compose up --build
```

## Available API endpoints

### POST /predict/

Upload an image file and provide the true digit label to get a prediction and save results.

Form-data parameters:

```params
file (file): Image file (e.g., PNG)

true_label (int): True digit label (0-9)
```

### GET /prediction-accuracy/

Retrieve prediction accuracy statistics from stored data.
Example response:

```json
{
  "total_predictions": 200,
  "correct_predictions": 178,
  "incorrect_predictions": 22,
  "accuracy_percent": 89.0
}
```
