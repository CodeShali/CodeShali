FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent/ ./agent/

# Resume is baked in at agent/myresume; optionally override via volume mount
EXPOSE 5000

CMD ["python", "-m", "agent.main"]
