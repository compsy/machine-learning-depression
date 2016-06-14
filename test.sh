#!/bin/bash
echo 'Running yapf'
PRE_DIFF=$(git diff)
yapf -r -i learner
yapf -r -i tests
POST_DIFF=$(git diff)

DIFF=$(diff <(echo "$PRE_DIFF") <(echo "$POST_DIFF"))
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
