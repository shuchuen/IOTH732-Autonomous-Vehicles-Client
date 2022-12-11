# syntax=docker/dockerfile:1

FROM python:3.9-slim-bullseye

WORKDIR /app

ENV VIRTUAL_ENV=/opt/.vehicles-client
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD [ "python", "vehicles_detect.py", "--modeldir=TFLite_model_bbd", "--output_path=/logs/processed_images", "--vin=auto-car-1"]
