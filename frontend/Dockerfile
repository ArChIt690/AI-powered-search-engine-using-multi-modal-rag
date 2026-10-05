# The frontend image: the Streamlit pages only (app.py for users, admin.py for admins; docker-compose.yml picks
# one). It reaches the backend over HTTP (API_URL) and imports nothing from it, so it needs just Streamlit and
# httpx (versions as in uv.lock).
FROM python:3.12-slim

RUN pip install --no-cache-dir streamlit==1.64.0 httpx==0.28.1

RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --chown=app:app app.py admin.py shared.py api_client.py ./
USER app

ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true"]
