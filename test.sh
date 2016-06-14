#!/bin/bash
yapf -r -i learner
DIFF=$(git diff)
if [ "$DIFF" != ""]
then
  # Something was wrong
  exit 1
fi
