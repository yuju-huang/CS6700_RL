#!/bin/bash

DIR=`dirname 0`
MODEL_ROOT=$DIR/trained_models/latest/
models=(${MODEL_ROOT}/training_lat0.5_util0.5/ ${MODEL_ROOT}/training_lat0.8_util0.2/ ${MODEL_ROOT}/training_lat0.2_util0.8/ ${MODEL_ROOT}/training_lat0.2_util0.8/_retrained/)

for model in ${models[@]}; do
    date; sudo $DIR/run.sh predict $model
done

