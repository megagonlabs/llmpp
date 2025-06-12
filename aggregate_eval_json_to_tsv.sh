#!/bin/bash

set -e

python llmpp/aggregate_json_to_tsv.py \
  sentence.gold sentence.content sentence.aligned sentence.correct_form \
  token.gold token.content token.aligned token.correct_form \
  token.correct_upos \
  token.correct_head \
  token.correct_head_deprel \
  - \
  "$@"
