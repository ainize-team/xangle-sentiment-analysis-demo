FROM python:3.7

COPY requirements.txt .

RUN apt-get update -y && \
    apt-get install -y python3-dev

RUN pip install --no-cache-dir --upgrade pip && \
    pip install -r requirements.txt

WORKDIR /app

COPY . .

EXPOSE 8051

CMD ["opyrator", "launch-ui", "sentiment_analysis_demo:xangle_sentiment_analysis"] 