From continuumio/miniconda3

RUN apt-get update \
    && apt-get install -y vim

RUN conda install -c conda-forge cyipopt ffmpeg \
    && conda install jupyter matplotlib \
    && pip install autograd parameterized nose pylint autopep8 bottle webtest

WORKDIR /BikeSim
ENV PYTHONPATH "${PYTHONPATH}:/BikeSim/bikesim"

CMD apps/webserver.py
