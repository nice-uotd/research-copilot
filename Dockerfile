FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app
COPY space_main.py gradio_app.py pyproject.toml README.md LICENSE ./
COPY seed_docs ./seed_docs
COPY eval ./eval

RUN mkdir -p /app/data /app/uploads

EXPOSE 7860

CMD ["python", "space_main.py"]
