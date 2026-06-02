FROM python:3.10-slim-bullseye

RUN python -m venv /opt/venv

WORKDIR /app

ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt  /temp/requirements.txt

RUN pip install -r /temp/requirements.txt

COPY . /app

CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
