#!/bin/bash

docker run -ti --runtime=nvidia --rm -e PYTHONIOENCODING=utf-8 \
-v `pwd`/source/:/source \
-v `pwd`/../dataset/wiki/:/dataset:ro \
ductricse/pytorch /bin/bash
