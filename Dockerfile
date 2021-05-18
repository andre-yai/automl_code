
FROM continuumio/miniconda3

RUN mkdir /app

# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]

COPY . /app

WORKDIR /app

# Create the environment:
RUN pip3 install fastapi uvicorn joblib numpy sklearn
RUN conda env create -f environment.yml 
# Activate the environment, and make sure it's activated:
# RUN conda activate autoML
# Make RUN commands use the new environment:

EXPOSE 8000:8000

RUN echo "alias l='ls -lah'" >> ~/.bashrc
RUN echo "source activate autoML" >> ~/.bashrc

# Setting these environmental variables is the functional equivalent of running 'source activate my-conda-env'
ENV CONDA_EXE /opt/conda/bin/conda
ENV CONDA_PREFIX /opt/conda/envs/connect
ENV CONDA_PYTHON_EXE /opt/conda/bin/python
ENV CONDA_PROMPT_MODIFIER (autoML)
ENV CONDA_DEFAULT_ENV autoML
ENV PATH /opt/conda/envs/connect/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

WORKDIR /app/src
ENV ConfigFile=./../models/Diabetes_Model.json
ENV ModelFolder=./../models
CMD [ "uvicorn", "main_modelServing:app", "--host", "0.0.0.0", "--port", "15400" ]

# Redis Dockerfile
#
# https://github.com/dockerfile/redis
#

#FROM redis
# COPY redis.conf /usr/local/etc/redis/redis.conf
# CMD [ "redis-server", "/usr/local/etc/redis/redis.conf" ]
# Define default command.
#CMD ["redis-server", "/etc/redis/redis.conf"]

# Expose ports.
#EXPOSE 6379