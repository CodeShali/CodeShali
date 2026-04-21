FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent/ ./agent/

# Resume is mounted at runtime: -v /path/to/resume.docx:/app/resume/resume.docx
RUN mkdir -p /app/resume

EXPOSE 5000

CMD ["python", "-m", "agent.main"]
