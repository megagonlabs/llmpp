#!/bin/bash

set -eu

for d in $@ ; do
  lang=${d#data/}
  lang=${lang%:*}
  for t in `ls templates/*.toml` ; do
    python -m llmpp.conllu_to_prompt --t ${t} --i ${d}/*.conllu --l ${lang} &
  done
done
wait
