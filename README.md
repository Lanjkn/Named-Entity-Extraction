# Named-Entity-Extraction
This is a Named Entity Extraction API, using the following models:

[NE Recognition - PT-BR](https://huggingface.co/cnmoro/ptt5-small-named-entity-recognition)

[NE Recognition - EN-US](https://huggingface.co/cnmoro/t5-small-named-entity-recognition)

Model creation credits goes to [Carlo Moro](https://huggingface.co/cnmoro)

# How to install
> docker build -t ne_recognition -f Dockerfile .

# How to run
> docker run -it -p 6363:6363 ne_recognition

# Using the Named Entity Recognition API
The api will be available at 0.0.0.0:6363, on the endpoint recognize.
> http://localhost:6363/recognize

Api Swagger is also available at /docs
> http://localhost:6363/docs


