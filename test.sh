#!/bin/bash
yapf -r -i learner
DIFF=$(git diff)
if [ "$DIFF" != "" ]
then
  # Something was wrong
  echo 'Yapf failed:'
  echo $DIFF
  exit 1
fi
echo 'Yapf ran succesfully'
