FROM python:3.5

RUN echo deb http://cran.rstudio.com/bin/linux/debian jessie-cran3/ >> /etc/apt/sources.list
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 06F90DE5381BA480

# Install dependencies
RUN apt-get -y update\
    && apt-get -y upgrade\
    && apt-get -y install r-base r-base-core\
                                  openmpi-bin openmpi-common libopenmpi-dev
    #&& apt-get -y install r-base r-base-core python3-dev python-dev pkg-config\
                                 #libatlas-base-dev liblapack-dev vim-nox\
                                  #openmpi-bin openmpi-common libopenmpi-dev
    #&& apt-get autoremove -y\
    #&& rm -rf /var/lib/apt/lists/*


# Copy files for installing python dependencies
COPY requirements.txt /tmp
COPY test-requirements.txt /tmp
COPY install-packages.R /tmp

WORKDIR /tmp
RUN pip install rpy2==2.8
RUN pip install -r requirements.txt
RUN pip install -r test-requirements.txt

# Install R files
RUN echo 'options(repos=structure(c(CRAN="http://cran.cnr.berkeley.edu/")))' > ~/.Rprofile
RUN R CMD BATCH install-packages.R

########################################
# Preferably add stuff after this line #
########################################

# Create a directory for the exports
VOLUME /exports

# Copy the data
COPY data /data
COPY run_on_docker /tmp

# Set env variables
ENV GITHUB_TOKEN=CHANGEMEINTHECORRECTTOKEN
ENV PYTHONPATH=.:/usr/local/bin/python:


# Create the directory to store the app in
RUN mkdir -p /usr/src

#ENV R_HOME=/usr/local/lib/R
#ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$R_HOME/lib
RUN mv /tmp/run_on_docker /usr/src

WORKDIR /usr/src
CMD [ "./run_on_docker"]
