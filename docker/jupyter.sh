#!/bin/bash

docker logs -f --timestamps $(docker run --runtime=nvidia -d -e PYTHONIOENCODING=utf-8 --name=auto_clf_jupyter --rm \
-v `pwd`/source/:/source \
-v `pwd`/../dataset:/dataset:ro \
-p 10000:10000 \
ductricse/pytorch /bin/bash -c "jupyter notebook --port=10000 --allow-root --ip=0.0.0.0")
