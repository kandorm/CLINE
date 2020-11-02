#!/usr/bin/env bash
DISK_DIR=.
DISK_CODE=.
DISK_DATA=$DISK_DIR/data

LOG_DIR=$DISK_DIR/log
OUTPUT_DIR=$DISK_DIR/model
CACHE_DIR=$DISK_DIR/cache

CONFIG_DIR=$DISK_CODE/config
TOKENIZE_DIR=$DISK_DATA/tokenize
DATA_DIR=$DISK_DATA/disk

if [ ! -d $LOG_DIR ]; then
  mkdir $LOG_DIR
fi

if [ ! -d $OUTPUT_DIR ]; then
  mkdir $OUTPUT_DIR
fi

if [ ! -d $CACHE_DIR ]; then
  mkdir $CACHE_DIR
fi

if [ ! -d $TOKENIZE_DIR ]; then
  mkdir $TOKENIZE_DIR
fi

CORPUS_NAME=enwiki_bookcorpus
MODEL_NAME=lecbert

CONFIG_PATH=$CONFIG_DIR/en/$MODEL_NAME-tiny-config.json
TOKENIZE_PATH=$TOKENIZE_DIR/$CORPUS_NAME
OUTPUT_PATH=$OUTPUT_DIR/$CORPUS_NAME-$MODEL_NAME-tiny
DATA_PATH=$DATA_DIR/$CORPUS_NAME-tiny-lec-disk


python3 -m torch.distributed.launch --nproc_per_node 8 --nnodes=1 --node_rank=0 $DISK_CODE/src/run.py \
    --model_type $MODEL_NAME \
    --output_dir $OUTPUT_PATH \
    --config_name $CONFIG_PATH \
    --tokenizer_name $TOKENIZE_PATH \
    --cache_dir $CACHE_DIR \
    --logging_dir $LOG_DIR \
    --train_data_file $DATA_PATH \
    --load_from_disk \
    --mlm \
    --mlm_probability 0.15 \
    --block_size 256 \
    --do_train \
    --prediction_loss_only \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 0.0001 \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 0.000001 \
    --max_steps 1000000 \
    --warmup_steps 10000 \
    --seed 12345 \
    --save_steps 1000 \
    --logging_steps 1000 \
    --save_total_limit 10
