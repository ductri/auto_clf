#!/bin/bash

docker logs -f --timestamps $(docker run -d --runtime=nvidia -e PYTHONIOENCODING=utf-8 --name="auto_clf_$(date +"%y-%m-%d_%H_%M_%S")" \
-v `pwd`/source/:/source \
-v `pwd`/../dataset/wiki/:/dataset:ro \
ductricse/pytorch /bin/bash -c "/source/scripts/train.sh")

