#!/bin/bash

set -eu

sampling_rate=$1
datasets=${@:2:($#-1)}

for d in ${datasets} ; do
  src_dir=data/${d}-r2.15
  dst_dir=data/${d}_${sampling_rate}-r2.15
  mkdir -p ${dst_dir}
  for s in train test dev ; do
    echo "${src_dir}/*${s}.jsonl > ${dst_dir}"
    cd ${src_dir}
    files=`ls *${s}.jsonl`
    cd - > /dev/null
    for f in ${files} ; do
      python -m llmpp.sampling ${sampling_rate} < ${src_dir}/${f} > ${dst_dir}/${f}
    done
  done
done
