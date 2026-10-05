FROM python:3.12-slim

# libgomp: OpenMP runtime XGBoost links against.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Pass the key at runtime:  docker run -p 8501:8501 --env-file .env ada
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
