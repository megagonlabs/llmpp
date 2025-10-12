#!/bin/bash

set -eu

d=$1
l=$2
for t in `ls templates/*.toml` ; do
  python -m llmpp.conllu_to_prompt --t ${t} --i ${d}/*.conllu --l ${l} &
done
wait
