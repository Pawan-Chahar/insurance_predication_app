# Insurance Prediction App

A full-stack Python application to predict insurance charges (or insurance risk/claim outcomes) using a machine learning model.  
This repository includes the backend API, model artifacts, and instructions to run locally, test, and deploy.



## 📌 Project Overview

This project provides a REST API and (optional) user interface for predicting insurance outcomes.  
The model is trained on insurance data and deployed via FastAPI (or Flask/Streamlit, depending on your implementation).

**Key Features**
- Machine Learning-powered insurance prediction
- REST API using FastAPI
- Clean project structure
- Model serialization & inference
- Easy deployment with Docker



## 🚀 Features

✔ Predict insurance cost / claim probability  
✔ API endpoint for single instance prediction  
✔ (Optional) Web UI to interact with the model  
✔ Modular code base for easy extension  
✔ Docker support for containerized deployment


## 🧠 Tech Stack

| Layer | Technology |
|-------|------------|
| Python | 3.8+ |
| API | FastAPI |
| ML | scikit-learn / XGBoost / any model |
| Serialization | pickle / joblib |
| Deployment | Docker, Uvicorn |
| Testing | Pytest (optional) |



## 🗂 Repository Structure







```text
insurance_predication_app/
│
├── model/                      # Serialized model artifacts
│   └── insurance_model.pkl
│
├── app/                        # Backend application source
│   ├── main.py                # FastAPI entry point
│   ├── schemas.py             # Pydantic schemas
│   ├── utils.py               # Helper functions
│   └── predict.py             # Prediction logic
│
├── data/                       # Dataset samples
│   └── insurance.csv
│
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker build
├── .dockerignore
├── README.md                  # This file
└── tests/                     # Unit tests
````



## 🔧 Installation – Local Setup

1. **Clone the repository**

```bash
git clone https://github.com/Pawan-Chahar/insurance_predication_app.git
cd insurance_predication_app
```

2. **Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```



## 🧪 Run the API

```bash
uvicorn app.main:app --reload
```

Open your browser or API tool and access:

```
http://127.0.0.1:8000/docs
```

This gives you interactive Swagger UI for API testing.



## 📌 API Endpoints

### **POST** `/predict`

Predict insurance outcome:

**Request:**

```json
{
    "age": 29,
    "bmi": 27.3,
    "smoker": "yes",
    "region": "southwest"
}
```

**Response:**

```json
{
  "prediction": 23145.67
}
```



## 🧠 Model Training (Optional)

If your repository contains training code:

```bash
python train_model.py --data data/insurance.csv
```

The script will produce a model file in `model/`.



## 📦 Docker Deployment

```bash
docker build -t insurance_pred_app .
docker run -p 8000:8000 insurance_pred_app
```


## 🧪 Testing

If tests are provided:

```bash
pytest
```

Perfect — below is a **production-ready, interview-grade setup** for your **Insurance Prediction App**, aligned with **FastAPI best practices**, **clean architecture**, and **local Docker development**.



# 1️⃣ Cleaned Project Structure (Recommended)

```text
insurance_prediction_app/
│
├── app/
│   ├── __init__.py
│   ├── main.py                # FastAPI entry point
│   ├── config.py              # Environment & settings
│   ├── schemas.py             # Pydantic request/response models
│   ├── router.py              # API routes
│   ├── service.py             # Business logic / prediction logic
│   └── model_loader.py        # Load ML model
│
├── model/
│   └── insurance_model.pkl    # Trained ML model
│
├── data/
│   └── insurance_sample.csv
│
├── tests/
│   └── test_predict.py
│
├── .env                       # Environment variables
├── .dockerignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── pyproject.toml (optional)
```

📌 **Why this structure is good**

* Clear separation of concerns
* Scales well for real production systems
* Matches FAANG-style backend expectations



# 2️⃣ Detailed API Schema (`schemas.py`)

```python
from pydantic import BaseModel, Field
from typing import Literal


class InsuranceRequest(BaseModel):
    age: int = Field(..., gt=0, description="Age of the customer")
    bmi: float = Field(..., gt=0, description="Body Mass Index")
    children: int = Field(..., ge=0, description="Number of children")
    smoker: Literal["yes", "no"]
    sex: Literal["male", "female"]
    region: Literal["southwest", "southeast", "northwest", "northeast"]


class InsuranceResponse(BaseModel):
    predicted_cost: float
```

📌 **Interview tip**
Using `Literal` gives **strong validation + auto Swagger docs**.



# 3️⃣ Example `.env` File

```env
# App config
APP_NAME=insurance-prediction-api
ENV=local
LOG_LEVEL=INFO

# Server
HOST=0.0.0.0
PORT=8000

# Model
MODEL_PATH=model/insurance_model.pkl
```

📌 **Best practice**

* Never hardcode paths or secrets
* `.env` is ignored via `.gitignore`



# 4️⃣ Supporting Files (Minimal but Clean)

## `config.py`

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str
    env: str
    log_level: str
    model_path: str

    class Config:
        env_file = ".env"


settings = Settings()
```



## `model_loader.py`

```python
import joblib
from app.config import settings


def load_model():
    return joblib.load(settings.model_path)
```



## `service.py`

```python
import pandas as pd
from app.model_loader import load_model

model = load_model()


def predict_insurance(data: dict) -> float:
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return float(prediction[0])
```


## `router.py`

```python
from fastapi import APIRouter
from app.schemas import InsuranceRequest, InsuranceResponse
from app.service import predict_insurance

router = APIRouter()


@router.post("/predict", response_model=InsuranceResponse)
def predict(payload: InsuranceRequest):
    result = predict_insurance(payload.model_dump())
    return InsuranceResponse(predicted_cost=result)
```



## `main.py`

```python
from fastapi import FastAPI
from app.router import router

app = FastAPI(title="Insurance Prediction API")

app.include_router(router)
```



# 5️⃣ Docker Setup

## `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```



## `.dockerignore`

```text
__pycache__
venv
.env
.git
```



# 6️⃣ Docker Compose for Local Development

```yaml
version: "3.9"

services:
  api:
    build: .
    container_name: insurance_api
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - .:/app
    restart: always
```



# 7️⃣ Run Locally (One Command)

```bash
docker-compose up --build
```

Open:

```
http://localhost:8000/docs
```



