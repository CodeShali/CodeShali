FROM python:3.11-slim

WORKDIR /app

# Persistent data directory (mount a volume here in production)
RUN mkdir -p /data

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent/ ./agent/

VOLUME ["/data"]
EXPOSE 5000

CMD ["python", "-m", "agent.main"]
