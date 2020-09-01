FROM python:3.8
RUN apt-get update
COPY . /main
WORKDIR /main
RUN pip install -U pip
RUN pip install -r requirements.txt
CMD python /main/src/pipeline_multi_model.py