# syntax=docker/dockerfile:1

FROM python:3.9-slim-bullseye

WORKDIR /app

ENV VIRTUAL_ENV=/opt/.vehicles-client
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD [ "python", "log_service.py", "--input_path=/logs/processed_images", "--api_path=http://192.168.2.10:8081/log"]
