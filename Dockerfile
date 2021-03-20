FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY ./app /app
COPY ./requirements.txt .

# install dependencies
RUN apt-get update 
RUN pip install -r requirements.txt