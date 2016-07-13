FROM python:3.5

RUN echo deb http://cran.rstudio.com/bin/linux/debian jessie-cran3/ >> /etc/apt/sources.list
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 06F90DE5381BA480
#RUN add-apt-repository ppa:marutter/rdev

RUN apt-get -y update\
      && apt-get -y upgrade\
      && apt-get -y install r-base libatlas-base-dev liblapack-dev \
      && apt-get -y autoremove

COPY requirements.txt /tmp
WORKDIR /tmp

RUN ln -s /box /root/.cache/pip
RUN pip install -r requirements.txt

RUN echo 'options(repos=structure(c(CRAN="http://cran.cnr.berkeley.edu/")))' > ~/.Rprofile

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
RUN mkdir exports
COPY install-packages.R /usr/src/app/
RUN R CMD BATCH install-packages.R

COPY learner /usr/src/app/learner
COPY data /usr/src/app/data
RUN ln -s /usr/src/app/exports /exports

#ENV R_HOME=/usr/local/lib/R
#ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$R_HOME/lib



WORKDIR /usr/src/app/learner
CMD [ "python", "main.py" ]
