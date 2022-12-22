FROM python:3.9

WORKDIR /

COPY trainer /trainer

RUN pip install cloudml-hypertune \
    pip install -r trainer/requirements.txt



ENTRYPOINT [ "python", "-m","trainer.train" ]