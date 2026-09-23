# Hugging Face Spaces (Docker SDK). Streamlit Community Cloud ignores this file.
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Spaces run as a non-root user
RUN useradd -m -u 1000 user && chown -R user /app
USER user

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
