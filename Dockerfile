# Use official Python image
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "app.app:app", "--bind", "0.0.0.0:8080", "--workers=2", "--timeout=900"]

EXPOSE 8080