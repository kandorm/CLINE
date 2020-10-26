#!/usr/bin/env bash
OUTPUT_DIR=.
CONFIG_DIR=./config
DATA_DIR=$OUTPUT_DIR/data
CACHE_DIR=$OUTPUT_DIR/cache
TOKENIZE_DIR=$DATA_DIR/tokenize
CORPUS_DIR=$DATA_DIR/corpus
DISK_DIR=$DATA_DIR/disk

if [ ! -d $DISK_DIR ]; then
  mkdir $DISK_DIR
fi

if [ ! -d $CACHE_DIR ]; then
  mkdir $CACHE_DIR
fi

if [ ! -d $TOKENIZE_DIR ]; then
  mkdir $TOKENIZE_DIR
fi

CORPUS_NAME=enwiki_bookcorpus
FILE=enwiki0000
MODEL_NAME=roberta

CORPUS_PATH=$CORPUS_DIR/$CORPUS_NAME/$FILE
CONFIG_PATH=$CONFIG_DIR/en/$MODEL_NAME-tiny-config.json
TOKENIZE_PATH=$TOKENIZE_DIR/$CORPUS_NAME
DATA_CACHE_PATH=$CACHE_DIR/$CORPUS_NAME-$FILE-tiny-train.arrow

PREPROCESS_BATCH_SIZE=1000
BLOCK_SIZE=256
PREPROCESS_NUM_PROCESS=1

python3 src/dataloader.py \
    --train_data_file $CORPUS_PATH \
    --cache_dir $CACHE_DIR \
    --config_name $CONFIG_PATH \
    --tokenizer_name $TOKENIZE_PATH \
    --block_size $BLOCK_SIZE \
    --lang en \
    --word_replace \
    --preprocess_batch_size $PREPROCESS_BATCH_SIZE \
    --preprocess_cache_file $DATA_CACHE_PATH \
    --preprocess_num_process $PREPROCESS_NUM_PROCESS \
    --preprocess_output_file $DISK_DIR/$CORPUS_NAME-$FILE-tiny-disk
