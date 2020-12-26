FROM python:3.8-slim

COPY my_ml_server/code/requirements.txt /root/my_ml_server/code/requirements.txt

RUN chown -R root:root /root/my_ml_server

WORKDIR /root/my_ml_server/code
RUN pip3 install -r requirements.txt

COPY my_ml_server/ ./
RUN chown -R root:root ./

ENV SECRET_KEY hello
ENV FLASK_APP run.py

RUN ["chmod", "+x", "code/run.py"]
CMD ["python3", "run.py"]