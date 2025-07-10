FROM python:3.11

WORKDIR /code

# Install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

# Copy app files
COPY ./app /code/app
COPY ./optimized_model.pkl /code/app/
COPY ./scaler.pkl /code/app/

# Use Flask instead of Uvicorn (this was for FastAPI)
CMD ["python", "app/app.py"]
