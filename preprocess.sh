#!/usr/bin bash
DATA_DIR=./data
CONFIG_DIR=./config
CACHE_DIR=./cache
TOKENIZE_DIR=$DATA_DIR/tokenize

if [ ! -d $CACHE_DIR ]; then
  mkdir $CACHE_DIR
fi

if [ ! -d $TOKENIZE_DIR ]; then
  mkdir $TOKENIZE_DIR
fi

CORPUS_NAME=enwiki_bookcorpus
MODEL_NAME=roberta

CORPUS_PATH=$DATA_DIR/$CORPUS_NAME
CONFIG_PATH=$CONFIG_DIR/en/$MODEL_NAME-tiny-config.json
TOKENIZE_PATH=$TOKENIZE_DIR/$CORPUS_NAME
DATA_CACHE_PATH=$CACHE_DIR/$CORPUS_NAME-train.arrow

PREPROCESS_BATCH_SIZE=500
BLOCK_SIZE=256
PREPROCESS_NUM_PROCESS=4

python3 $DISK_CODE/src/dataloader.py \
    --train_data_file $CORPUS_PATH \
    --cache_dir $CACHE_DIR \
    --config_name $CONFIG_PATH \
    --tokenizer_name $TOKENIZE_PATH \
    --block_size $BLOCK_SIZE \
    --preprocess_batch_size $PREPROCESS_BATCH_SIZE \
    --preprocess_cache_file $DATA_CACHE_PATH \
    --preprocess_num_process $PREPROCESS_NUM_PROCESS \
    --preprocess_output_file $DATA_DIR/$CORPUS_NAME-disk
