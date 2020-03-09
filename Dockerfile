FROM python:3.8-slim-buster

WORKDIR /app

RUN apt-get update \
    && apt-get install -y build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements-serve.txt ./requirements-serve.txt
RUN pip3 install --no-cache-dir -r requirements-serve.txt \
    && pip3 install uwsgi

COPY . .

EXPOSE 8000
CMD ["uwsgi", \
    "--socket", "0.0.0.0:8000", \
    "--protocol", "uwsgi", \
    "--wsgi", "run_server:app" ]