FROM nvcr.io/nvidia/pytorch:22.05-py3

WORKDIR /root

RUN apt-get update
ADD requirements.txt .
RUN pip install -r requirements.txt

EXPOSE 8888
EXPOSE 6006

CMD ["bash"]