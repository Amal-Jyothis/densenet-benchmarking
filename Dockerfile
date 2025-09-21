FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime
# FROM python:3.10-slim
WORKDIR /benchmark
COPY . /benchmark
COPY requirements.txt .
RUN pip install -r requirements.txt

