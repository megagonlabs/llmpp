#!/bin/bash

set -eu

template_path=$1
for d in ${@:2:($#-1)} ; do
  lang=${d#data/}
  lang=${lang%@*}
  for t in `ls ${template_path}` ; do
    python -m llmpp.conllu_to_prompt --t ${t} --i ${d}/*.conllu --l ${lang} &
  done
done
wait
