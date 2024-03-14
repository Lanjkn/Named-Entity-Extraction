FROM python:3.11

WORKDIR /app

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY . .

RUN wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz

ENTRYPOINT ["uvicorn", "named_entity_recognition_api:app", "--port", "6363", "--workers", "2", "--host", "0.0.0.0"]

EXPOSE 6363