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
  mkdir $CACHE_DIR
fi

CORPUS_NAME=enwiki_bookcorpus
MODEL_NAME=roberta

CORPUS_PATH=$DATA_DIR/$CORPUS_NAME
CONFIG_PATH=$CONFIG_DIR/en/$MODEL_NAME-base-config.json
TOKENIZE_PATH=$TOKENIZE_DIR/$CORPUS_NAME
OUTPUT_PATH=$OUTPUT_DIR/$CORPUS_NAME
DATA_CACHE_PATH=$CACHE_DIR/$CORPUS_NAME-train.arrow

PREPROCESS_BATCH_SIZE=1000
BLOCK_SIZE=512
PREPROCESS_NUM_PROCESS=8
MLM_PROBABILITY=0.15

python3 src/dataloader.py \
    --train_data_file $CORPUS_PATH \
    --cache_dir $CACHE_DIR \
    --config_name $CONFIG_PATH \
    --tokenizer_name $TOKENIZE_PATH \
    --block_size $BLOCK_SIZE \
    --preprocess_batch_size $PREPROCESS_BATCH_SIZE \
    --preprocess_cache_file $DATA_CACHE_PATH \
    --preprocess_num_process $PREPROCESS_NUM_PROCESS

python3 src/run.py \
    --model_type $MODEL_NAME \
    --output_dir $OUTPUT_PATH \
    --config_name $CONFIG_PATH \
    --tokenizer_name $TOKENIZE_PATH \
    --cache_dir $CACHE_DIR \
    --train_data_file $CORPUS_PATH \
    --mlm \
    --mlm_probability $MLM_PROBABILITY \
    --block_size $BLOCK_SIZE \
    --preprocess_batch_size $PREPROCESS_BATCH_SIZE \
    --preprocess_cache_file $DATA_CACHE_PATH \
    --preprocess_num_process $PREPROCESS_NUM_PROCESS \
    --do_train \
    --prediction_loss_only \
    --per_device_train_batch_size 8000 \
    --gradient_accumulation_steps 1 \
    --learning_rate 0.0006 \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 0.000001 \
    --max_grad_norm 0.0 \
    --max_steps 500000 \
    --warmup_steps 24000 \
    --seed 12345 \
    --save_steps 1000 \
    --save_total_limit 3 \
    --overwrite_output_dir
