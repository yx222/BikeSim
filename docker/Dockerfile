From continuumio/miniconda3

RUN conda install -c conda-forge cyipopt
RUN conda install jupyter

RUN pip install autograd
RUN conda install matplotlib
RUN conda install -c conda-forge ffmpeg

WORKDIR /BikeSim
CMD python3 webserver.py
