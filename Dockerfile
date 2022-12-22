FROM python:3.9

RUN apt-get update
RUN mkdir app
COPY requirements.txt /app

RUN pip install --upgrade pip \
    pip install -r /app/requirements.txt