# Learning Emotions
[![CircleCI](https://circleci.com/gh/compsy/ICPE_machine_learning_workgroup.svg?style=svg&circle-token=4e926b5d1a43abc4e98c0aa227695a50340848a3)](https://circleci.com/gh/compsy/ICPE_machine_learning_workgroup)

This is the repository for the ICPE machine learning workgroup. In this readme we present how one can setup the software and run the analysis.

## Intalling
The procedure to run the software is as follows. There exists a `setup.sh` file, but that's still in development, and following the next steps probably gives a better result

### 1. Install the dependencies
First the dependencies used by the application need to be installed. Open a terminal, clone the project, and `cd` to the cloned directory. Make sure you have python 3.5 installed. Then, depending on your preferences, create a virtual environment to save the dependencies in. Note that this is a Python 3.5 project, and we need to use a Python 3.x virtual environment.

```
python3 -m venv venv
source venv/bin/activate
```

Your terminal should now show that you are using the `venv` virtual environment. The final step is to install the dependencies. The easiest way to installing the dependencies is by using pip:

```
pip install -r requirements.txt
```

### 2. Initializeing the data and cache
The data used for the present project is provided by NESDA. The easiest method to get the data in the project is by simlinking to the location where the data is stored. In case of Compsy development machines the following lines suffice:

```
ln -s ~/vault/NESDA/SPSS data
```

After linking the data, it is important to create a cache directory to store the data and model caches.

```
mkdir -p cache/mlmodels
mkdir exports
```

### 3. Setting up AWS credentials
In the current setup the whole system is based on exchanging data with AWS. ML models are created in the application, dumped to disk, and uploaded to AWS so multiple clients could work on the same project. The package uses a python package to perform these data uploads, and these use the following env variables:
  
```
AWS_ACCESS_KEY_ID=CHANGEMEINTHECORRECTTOKEN
AWS_SECRET_ACCESS_KEY=CHANGEMEINTHECORRECTTOKEN
```

### 4. Running the software
To test whether everything works, we can now run the application. Because the design of the application is built in such a way that it can potentially be distributed over a number of machines, there are a number of different configurations one could use to start the analysis. The first step is to split the data in a test and training set. The test set is only used for evaluating the algorithm after it was trained on the training set. This training set is internally used as a cross-validation set. Creating the set can be done as follows:

```
python main.py -t createset -f -p -n
```

In this case, `-t` specifies the part of the application to run, `-f` specifies the use of feature selection, `-p` allows the use of polynomial features, and `-n` removes previous cached files.

The next step is to actually train the algorithms on these created datasets. This can be done using the following command:

```
python main.py -t train
```

What this steps does is run all of the models specified in `driver.py` and upload the fitted models to S3. After this step has completed, we can retrieve the results from S3 using the following command:

```
python main.py -t evaluate
```

Which exports the output of the project.

## Docker
Apart from installing everything locally, one could also run the application in Docker using the following command:
```
docker run --rm -it \
  -e "AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID"\
  -e "AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY"\
  --name ml frbl/icpe-machine-learning
```

The docker image can be updated by changing the code, and running the `build` script.









