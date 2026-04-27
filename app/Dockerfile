FROM python:3.11-slim

# php-cli est requis par validate_php_syntax() pour `php -l`
RUN apt-get update \
 && apt-get install -y --no-install-recommends php-cli \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=600 -r requirements.txt

COPY . .

# On lance l'API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]