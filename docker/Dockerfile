FROM python:3.5

RUN echo deb http://cran.rstudio.com/bin/linux/debian jessie-cran3/ >> /etc/apt/sources.list
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 06F90DE5381BA480
#RUN add-apt-repository ppa:marutter/rdev

RUN apt-get -y update\
    && apt-get -y upgrade\
    && apt-get -y install r-base python3-dev python-dev pkg-config libatlas-base-dev liblapack-dev vim-nox\
                          openmpi-bin openmpi-common libopenmpi-dev

COPY requirements.txt /tmp
COPY test-requirements.txt /tmp
WORKDIR /tmp
RUN pip install -r requirements.txt
RUN pip install -r test-requirements.txt

########################################
# Preferably add stuff after this line #
########################################

ENV GITHUB_TOKEN=CHANGEMEINTHECORRECTTOKEN
ENV PYTHONPATH=.:/usr/local/bin/python:

RUN echo 'options(repos=structure(c(CRAN="http://cran.cnr.berkeley.edu/")))' > ~/.Rprofile

# Create the directory to store the app in
RUN mkdir -p /usr/src
WORKDIR /usr/src
RUN git clone https://$GITHUB_TOKEN:x-oauth-basic@github.com/compsy/ICPE_machine_learning_workgroup.git app 
WORKDIR /usr/src/app

RUN R CMD BATCH install-packages.R

# Create a directory for the exports
RUN ln -s /usr/src/app/exports /exports
VOLUME /exports


# Copy the data
COPY data /usr/src/app/data

# Copy the actual application
#COPY learner /usr/src/app/learner
#COPY tests /usr/src/app/tests


#ENV R_HOME=/usr/local/lib/R
#ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$R_HOME/lib

WORKDIR /usr/src/app
CMD [ "pytest", "tests"]
