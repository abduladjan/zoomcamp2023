FROM python:3.9.18-slim

RUN pip install pipenv

WORKDIR /app
COPY["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY["predict.py", "model_regressor.bin",=.5=+/.2+%
 ==="./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]