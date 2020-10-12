#!/usr/bin bash
DATA_DIR=./data
LOG_DIR=./log
CONFIG_DIR=./config
OUTPUT_DIR=./model
CACHE_DIR=./cache
TOKENIZE_DIR=$DATA_DIR/tokenize

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
MODEL_NAME=roberta

CONFIG_PATH=$CONFIG_DIR/en/$MODEL_NAME-tiny-config.json
TOKENIZE_PATH=$TOKENIZE_DIR/$CORPUS_NAME
OUTPUT_PATH=$OUTPUT_DIR/$CORPUS_NAME-tiny
DATA_PATH=$DATA_DIR/$CORPUS_NAME-tiny-disk


python3 -m torch.distributed.launch --nproc_per_node 8 $DISK_CODE/src/run.py \
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
    --per_device_train_batch_size 64 \
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
