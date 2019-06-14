#!/bin/bash

docker logs -f --timestamps $(docker run --rm -d -e PYTHONIOENCODING=utf-8 --name="auto_clf_download" \
-v `pwd`/source:/source \
ductricse/pytorch /bin/bash -c "/source/scripts/download.sh")
