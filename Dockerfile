# syntax=docker/dockerfile:1

FROM python:3.7-slim-buster

WORKDIR /code 
COPY . .
RUN pip install -r requirements.txt

CMD ["coverage", "run", "test.py"]
