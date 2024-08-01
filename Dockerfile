FROM nvcr.io/nvidia/pytorch:24.07-py3-igpu

WORKDIR /root

RUN apt-get update
ADD requirements.txt .
RUN pip install -r requirements.txt

EXPOSE 8888
EXPOSE 6006

CMD ["bash"]