FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY ./app /app
COPY ./requirements.txt .
WORKDIR /app/

# install dependencies
RUN apt-get update 
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
