#!/bin/bash

set -eu

output_base=$1  # like ./data
revision=$2  # like r2.17
for long_name in ${@:3:($#-2)} ; do  # like UD_Japanese-GSD
    echo ${long_name}:${revision}
    tempfile=$(mktemp)
    trap 'rm -f "'${tempfile}'"; exit' EXIT
    curl -s -f -o "${tempfile}" "https://github.com/UniversalDependencies/${long_name}/"
    file_name=`egrep -m 1 -o '"[^-_"]+_[^-"]+-ud-test.conllu"' ${tempfile} | head -n 1`
    if [ -n ${file_name} ] ; then
        short_name=${file_name#\"}
        short_name=${short_name%-ud-test.conllu\"}
    else
        echo '*-ud-test.conllu not found'
        continue
    fi
    lang=${long_name#UD_}
    lang=${lang%-*}
    output_dir=./${output_base}/${lang}@${short_name}-${revision}
    mkdir -p ${output_dir}
    for s in train dev test ; do
        file_name=${short_name}-ud-${s}.conllu
        set +e
        curl -s -f -o ${output_dir}/${file_name} https://raw.githubusercontent.com/UniversalDependencies/${long_name}/refs/tags/${revision}/${file_name}
        if [ $? -eq 0 ]; then
            echo ${output_dir}/${file_name} created
        else
            echo ${file_name} not found
        fi
        set -e
    done
done
