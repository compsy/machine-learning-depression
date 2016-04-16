#!/bin/bash
rm -rf data
default=~/vault/NESDA/SPSS

echo "Please provide the location of the NESDA data directory ($default)"
read input_variable
if [ "$input_variable" = "" ]; then
  input_variable=$default
fi
ln -s $input_variable data
echo 'Linked data directory'

echo 'Seting up Virtual env, assuming Python 3.5'
pyvenv-3.5 venv
echo Now run:
tput setaf 1; echo "source venv/bin/activate"
tput setaf 1; echo "pip3.5 install -r requirements.txt"
