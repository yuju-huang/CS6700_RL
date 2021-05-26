#!/bin/bash

if [[ $(/usr/bin/id -u) -ne 0 ]]; then
    echo "Please run as root"
    exit
fi

if [ -z $1 ]; then
    echo "arg1 as (train | retrain | predict)"
    exit
fi

TIME=`date +%F_%H-%M-%S`
DIR=`dirname 0`
WORKLOAD_ROOT=/home/yh885/TailBench/xapian/

if [[ "$1" == "retrain" ]] || [[ "$1" == "predict" ]]; then
    if [ -z $2 ]; then
        echo "(retrain | predict) need arg2 as model path"
        exit
    fi
    MODEL_PATH=$2
else
    MODEL_PATH=$DIR/trained_models/$TIME
    mkdir -p $MODEL_PATH
fi

LOGS_PATH=$DIR/logs/$TIME
mkdir -p $LOGS_PATH

network_run() {
echo "-----" $@ "-----"
mode=$1
workload=$2
lat_weight=$3
util_weight=$4
p99_qps=$5
if [ -z $6 ]; then
    if [[ $mode != "train" ]]; then
        echo "Must specify model path for (predict | retrain)"
    fi
    model_path=${MODEL_PATH}/training_lat${lat_weight}_util${util_weight}/
else
    model_path=$6/training_lat${lat_weight}_util${util_weight}/
fi
log_file=${model_path}/${mode}_lat${lat_weight}_util${util_weight}_$workload.log

echo "Model path: "$model_path
mkdir -p $model_path
echo "Write log to "$log_file
rm $log_file
date > $log_file
sudo ipcrm --all=msg
#echo "sudo python3 train.py $mode $model_path $WORKLOAD_ROOT/$workload $lat_weight $util_weight $p99_qps #>> $log_file 2>&1"
sudo python3 train.py $mode $model_path $WORKLOAD_ROOT/$workload $lat_weight $util_weight $p99_qps $log_file #2>&1
if [[ $mode == "predict" ]]; then
    cat $log_file | grep timestamp > ${log_file}.parse
fi
}

lat_weights=(8)
#lat_weights=(1 2 5 8 9)
#lat_weights=(2 5 8)
p99_qps=10
do_train() {
workload="workload_fix4s_20s.dec"
for lat_weight in ${lat_weights[@]}; do
    util_weight=$((10 - lat_weight))
    lat_weight_p=`echo 'print('$lat_weight'/10)' | python3`
    util_weight_p=`echo 'print('$util_weight'/10)' | python3`
    network_run "train" $workload $lat_weight_p $util_weight_p $p99_qps
done
}

do_retrain() {
workload="workload_fix4s_20s.dec"
for lat_weight in ${lat_weights[@]}; do
    util_weight=$((10 - lat_weight))
    lat_weight_p=`echo 'print('$lat_weight'/10)' | python3`
    util_weight_p=`echo 'print('$util_weight'/10)' | python3`
    network_run "retrain" $workload $lat_weight_p $util_weight_p $p99_qps $MODEL_PATH
done
}

do_predict() {
workload="workload_fix4s_20s.dec"
#workload="workload_180s.dec"
for lat_weight in ${lat_weights[@]}; do
    util_weight=$((10 - lat_weight))
    lat_weight_p=`echo 'print('$lat_weight'/10)' | python3`
    util_weight_p=`echo 'print('$util_weight'/10)' | python3`
    network_run "predict" $workload $lat_weight_p $util_weight_p $p99_qps $MODEL_PATH
done
}

if [[ "$1" == "train" ]]; then
    echo "do_train"
    do_train
elif [[ "$1" == "retrain" ]]; then
    echo "do_retrain"
    do_retrain
elif [[ "$1" == "predict" ]]; then
    echo "do_predict"
    do_predict
else
    echo "Invalid mode "$1
fi
