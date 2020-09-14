#!/usr/bin bash
DATA_DIR=./data
LOG_DIR=./log
TOKENIZE_DIR=$DATA_DIR/tokenize

CORPUS_NAME=enwiki_bookcorpus

CORPUS_PATH=$DATA_DIR/$CORPUS_NAME
TOKENIZE_PATH=$TOKENIZE_DIR/$CORPUS_NAME
LOG_PATH=$LOG_DIR/$CORPUS_NAME

if [ ! -d $LOG_DIR ]; then
  mkdir $LOG_DIR
fi

if [ ! -d $TOKENIZE_DIR ]; then
  mkdir $TOKENIZE_DIR
fi

nohup python3 preprocess/tokenizer_train.py \
    --input_path $CORPUS_PATH \
    --output_path $TOKENIZE_PATH \
    --bytelevel \
    --prefix_space \
    --trim_offsets \
    --lowercase \
    --normalizer nfkc \
    --vocab 52000 \
    --minfreq 2 \
    > $LOG_PATH 2>&1 &
