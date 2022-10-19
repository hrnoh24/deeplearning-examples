#!/bin/bash

DATA_DIR="$1"

ARGS=""
ARGS+=" -it"
ARGS+=" --rm"
ARGS+=" --gpus all --ipc host"
ARGS+=" -p 6006:6006"
ARGS+=" -p 8888:8888"
ARGS+=" -v ${PWD}:/workspace"
[ "${DATA_DIR}" != "" ] && ARGS+=" -v ${DATA_DIR}:/data"

docker run ${ARGS} fobe