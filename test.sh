#!/bin/bash
echo 'Running yapf'
yapf -r -i learner
yapf -r -i tests

DIFF=$(git diff --name-only | grep '\.py')
if [ "$DIFF" != "" ]
then
  # Something was wrong
  echo 'Yapf failed:'
  echo $DIFF
  exit 1
fi
echo 'Yapf ran succesfully'

echo 'Running TOX'
tox
