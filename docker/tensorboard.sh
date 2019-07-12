#!/bin/bash

docker logs -f --timestamps $(docker run --rm -d -e PYTHONIOENCODING=utf-8 --name="auto_clf_tb" \
-v `pwd`/source:/source \
-p 10001:6006 \
ductricse/pytorch /bin/bash -c "/source/scripts/tensorboard.sh")
