#!/bin/bash

if [[ $(/usr/bin/id -u) -ne 0 ]]; then
    echo "Please run as root"
    exit
fi

if [ -z $1 ]; then
    echo "arg1 as (train | predict)"
    exit
fi

DIR=`dirname 0`
WORKLOAD_ROOT=/home/yh885/TailBench/xapian/
MODEL_PATH=$DIR/trained_models
LOGS_PATH=$DIR/logs

network_run() {
echo "-----" $@ "-----"
mode=$1
workload=$2
lat_weight=$3
util_weight=$4
p99_qps=$5
model_name=training_lat${lat_weight}_util${util_weight}.model
log_file=${LOGS_PATH}/${mode}_lat${lat_weight}_util${util_weight}_$workload.log

echo "Write log to "$log_file
rm $log_file
date > $log_file
sudo ipcrm --all=msg
sudo python3 train.py $mode $MODEL_PATH/$model_name $WORKLOAD_ROOT/$workload $lat_weight $util_weight $p99_qps >> $log_file 2>&1
}

#lat_weights=(5)
#lat_weights=(1 2 5 8 9)
lat_weights=(2 5 8)
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

do_predict() {
#workload="workload_fix4s_20s.dec"
workload="workload_180s.dec"
for lat_weight in ${lat_weights[@]}; do
    util_weight=$((10 - lat_weight))
    lat_weight_p=`echo 'print('$lat_weight'/10)' | python3`
    util_weight_p=`echo 'print('$util_weight'/10)' | python3`
    network_run "predict" $workload $lat_weight_p $util_weight_p $p99_qps
done
}

if [[ "$1" == "train" ]]; then
    echo "do_train"
    do_train
elif [[ "$1" == "predict" ]]; then
    echo "do_predict"
    do_predict
else
    echo "Invalid mode "$1
fi
