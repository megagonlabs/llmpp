#!/bin/bash

set -e

LONG_NAME=$1  # like UD_Japanese-GSD
SHORT_NAME=$2  # like ja_gsd
REVISION=$3  # like r2.15
OUTPUT_DIR=$4
if [ -z "$OUTPUT_DIR" ]; then  # like ./data/ja_gsd-r2.15/
    OUTPUT_DIR=./data/$SHORT_NAME-$REVISION
fi

mkdir -p $OUTPUT_DIR

for SET in train dev test ; do
    FILE_NAME=$SHORT_NAME-ud-$SET.conllu
    curl -o $OUTPUT_DIR/$FILE_NAME https://raw.githubusercontent.com/UniversalDependencies/$LONG_NAME/refs/tags/$REVISION/$FILE_NAME
done
