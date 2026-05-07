FROM python:3.10.19-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV API_URL=http://localhost:8000

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8000
EXPOSE 8501

CMD sh -c "uvicorn api_fastapi:app --host 0.0.0.0 --port 8000 & streamlit run api_streamlit.py --server.address 0.0.0.0 --server.port 8501"