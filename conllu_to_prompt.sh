#!/bin/bash

set -eu

d=$1
l=$2
python -m llmpp.conllu_to_prompt --t templates/en-*.toml                --i ${d}/*.conllu --l ${l} &
python -m llmpp.conllu_to_prompt --t templates/en-linearized-step*.toml --i ${d}/*.conllu --l ${l} --s sa  --r " " &
python -m llmpp.conllu_to_prompt --t templates/en-linearized-step*.toml --i ${d}/*.conllu --l ${l} --s sa2 --r "  " &
