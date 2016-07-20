FROM python:3.5

RUN echo deb http://cran.rstudio.com/bin/linux/debian jessie-cran3/ >> /etc/apt/sources.list
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 06F90DE5381BA480
#RUN add-apt-repository ppa:marutter/rdev

RUN apt-get -y update\
    && apt-get -y upgrade\
    && apt-get -y install r-base python3-dev python-dev pkg-config libatlas-base-dev liblapack-dev

COPY requirements.txt /tmp
WORKDIR /tmp
RUN pip install -r requirements.txt

########################################
# Preferably add stuff after this line #
########################################

RUN echo 'options(repos=structure(c(CRAN="http://cran.cnr.berkeley.edu/")))' > ~/.Rprofile

# Create the directory to store the app in
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Create a directory for the exports
RUN mkdir exports
RUN ln -s /usr/src/app/exports /exports
VOLUME /exports

COPY install-packages.R /usr/src/app/
RUN R CMD BATCH install-packages.R

# Copy the data
COPY data /usr/src/app/data

# Copy the actual application
COPY learner /usr/src/app/learner

#ENV R_HOME=/usr/local/lib/R
#ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$R_HOME/lib

WORKDIR /usr/src/app/learner
CMD [ "python", "main.py | tee ../exports/output.txt" ]
