#!/bin/bash
# run this script as:
# > source prepare_pythonpath.sh

export PYTHONPATH=$(pwd)/insightface:$(pwd):$(pwd)/privacy:$PYTHONPATH
