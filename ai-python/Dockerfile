FROM python:3.11-slim
WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY ./app /app/app
COPY ./scripts /app/scripts

EXPOSE 5000

# Run uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
