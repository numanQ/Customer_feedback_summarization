FROM python:3.10.12

WORKDIR /opt
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ .

EXPOSE 8888

ENTRYPOINT [ "python", "main.py" ]
