FROM python:3.10-slim

WORKDIR /app

COPY backend ./backend
COPY models ./models
COPY data ./data

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir fastapi uvicorn numpy pandas scikit-learn joblib shap matplotlib tensorflow==2.15 python-multipart xgboost

EXPOSE 8000

CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]