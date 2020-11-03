#!/usr/bin bash
DISK_DATA=./data/glue_data
MODEL_DIR=./model
OUTPUT_DIR=./output

if [ ! -d $OUTPUT_DIR ]; then
  mkdir $OUTPUT_DIR
fi

MODEL_PATH=$MODEL_DIR/enwiki_bookcorpus
TASK_NAME=CoLA  # CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE, WNLI
MODEL_TYPE=lecbert

python3 $DISK_CODE/src/run_glue.py \
  --model_name_or_path $MODEL_PATH \
  --model_type $MODEL_TYPE \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $DISK_DATA/$TASK_NAME \
  --max_seq_length 256 \
  --per_device_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir $OUTPUT_DIR/$TASK_NAME \
  --overwrite_output_dir
