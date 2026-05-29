FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/raw data/processed models results

ENV DATA_PATH=./data
ENV RAW_DATA_PATH=./data/raw
ENV PROCESSED_DATA_PATH=./data/processed
ENV MODEL_PATH=./models
ENV RESULTS_PATH=./results

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import sys, os; sys.exit(0 if os.path.exists('/app/src/prepare_data.py') else 1)"

CMD ["sh", "-c", \
    "python src/prepare_data.py && \
     python src/train_model.py && \
     python src/generate_explanation.py --user_id 1 --item_id 2 && \
     python src/evaluate_explanations.py"]
