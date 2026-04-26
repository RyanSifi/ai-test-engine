FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=600 -r requirements.txt

COPY . .

# On lance l'API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]