FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

RUN useradd --create-home --uid 1000 enhancer \
 && chown -R enhancer:enhancer /app
USER enhancer

EXPOSE 8080

CMD ["python", "-m", "app.main"]
