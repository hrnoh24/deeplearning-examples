#!/bin/bash

DATA_DIR="$1"

ARGS=""
ARGS+=" -it"
ARGS+=" --rm"

# Check if Nvidia GPU is available
if [[ $(lshw -C display | grep vendor) =~ Nvidia ]]; then
    echo "[*] Nvidia GPU found. Running on GPU."
    ARGS+=" --gpus all --ipc host"
else
    echo "[*] No Nvidia GPU found. Running on CPU."
fi

ARGS+=' --shm-size=2g'
ARGS+=" -p 6006:6006"
ARGS+=" -p 8888:8888"
ARGS+=" -v ${PWD}:/root"
[ "${DATA_DIR}" != "" ] && ARGS+=" -v ${DATA_DIR}:/root/data"

docker run ${ARGS} fobe