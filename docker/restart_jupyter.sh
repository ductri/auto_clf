#!/bin/bash

docker stop auto_clf_jupyter
sleep 5

./docker/jupyter.sh
