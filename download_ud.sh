#!/bin/bash

set -eu

LONG_NAME=$1  # like UD_Japanese-GSD
SHORT_NAME=$2  # like ja_gsd
REVISION=$3  # like r2.15
if [[ $# -ge 4 ]] ; then
    OUTPUT_DIR=$4
else
    LANGUAGE=${LONG_NAME#UD_}
    LANGUAGE=${LANGUAGE%-*}
    OUTPUT_DIR=./data/${LANGUAGE}@${SHORT_NAME}-${REVISION}
fi

mkdir -p ${OUTPUT_DIR}

for SET in train dev test ; do
    FILE_NAME=${SHORT_NAME}-ud-${SET}.conllu
    curl -o ${OUTPUT_DIR}/${FILE_NAME} https://raw.githubusercontent.com/UniversalDependencies/${LONG_NAME}/refs/tags/${REVISION}/${FILE_NAME}
done
