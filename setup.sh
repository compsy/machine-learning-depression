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
