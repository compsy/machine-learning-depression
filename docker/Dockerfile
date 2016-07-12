FROM python:3.5

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get -y install r-base libatlas-base-dev liblapack-dev

COPY learner /usr/src/app/learner
COPY data /usr/src/app/data
COPY requirements.txt /usr/src/app/
COPY install-packages.R /usr/src/app/

RUN echo 'options(repos=structure(c(CRAN="http://cran.cnr.berkeley.edu/")))' > ~/.Rprofile
RUN R CMD BATCH install-packages.R

ENV R_HOME=/usr/local/lib/R
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$R_HOME/lib

RUN pip install -r requirements.txt


CMD [ "python", "learner/main.py" ]
