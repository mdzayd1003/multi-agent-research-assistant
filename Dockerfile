FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY research_assistant ./research_assistant
COPY sample_corpus ./sample_corpus

ENV RA_LLM_BACKEND=ollama \
    RA_OLLAMA_HOST=http://ollama:11434 \
    RA_INDEX_DIR=/app/.cache/index \
    PYTHONUNBUFFERED=1

RUN python -m research_assistant ingest sample_corpus

EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "research_assistant/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
