FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY models ./models
COPY service ./service

EXPOSE 9696

CMD ["gunicorn", "--bind=0.0.0.0:9696", "service.app:app"]
