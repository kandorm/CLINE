import os
from datasets import load_dataset, Dataset
from typing import Optional
from dataclasses import dataclass, field

from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizer,
    MODEL_WITH_LM_HEAD_MAPPING
)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    plm_probability: float = field(
        default=1 / 6,
        metadata={
            "help": "Ratio of length of a span of masked tokens to surrounding context length for permutation language modeling."
        },
    )
    max_span_length: int = field(
        default=5, metadata={"help": "Maximum length of a span of masked tokens for permutation language modeling."}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    preprocess_batch_size: int = field(
        default=1000, metadata={"help": "Number of examples per batch provided to preprocess function."}
    )
    preprocess_cache_file: Optional[str] = field(
        default=None, metadata={"help": "Provide the name of a cache file to use to store the results of the computation instead of the automatically generated cache file name."}
    )
    preprocess_num_process: int = field(
        default=16, metadata={"help": "Number of processes for multiprocessing."}
    )
    load_from_disk: bool = field(
        default=False, metadata={"help": "Load dataset from disk."}
    )
    preprocess_output_file: Optional[str] = field(
        default=None, metadata={"help": "Path to preprocess dataset."}
    )


def get_dataset(
    args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizer,
    evaluate: bool = False,
    cache_dir: Optional[str] = None,
):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.load_from_disk:
        return Dataset.load_from_disk(file_path)

    if os.path.isdir(file_path):
        file_names = os.listdir(file_path)
        file_path = [os.path.join(file_path, fn) for fn in file_names]

    dataset = load_dataset("src/text.py", data_files=file_path, split="train", cache_dir=cache_dir, ignore_verifications=True)

    if args.line_by_line:
        #return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)
        dataset = dataset.map(lambda ex: tokenizer(ex["text"], add_special_tokens=True,
                                                   truncation=True, max_length=args.block_size),
                              batched=True,
                              batch_size=args.preprocess_batch_size,
                              writer_batch_size=args.preprocess_batch_size,
                              remove_columns=dataset.column_names,
                              load_from_cache_file=True,
                              cache_file_name=args.preprocess_cache_file,
                              num_proc=args.preprocess_num_process)

    else:
        # return TextDataset(
        #     tokenizer=tokenizer,
        #     file_path=file_path,
        #     block_size=args.block_size,
        #     overwrite_cache=args.overwrite_cache,
        #     cache_dir=cache_dir,
        # )
        def lines_to_block(examples):
            outputs = []
            block_size = args.block_size - tokenizer.num_special_tokens_to_add(pair=False)
            lines = examples['text']
            text = "\n".join(lines)
            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
            for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                outputs.append(
                    tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
                )
            # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.
            return {'input_ids': outputs}

        dataset = dataset.map(lines_to_block,
                              batched=True,
                              batch_size=args.preprocess_batch_size,
                              writer_batch_size=args.preprocess_batch_size,
                              remove_columns=dataset.column_names,
                              load_from_cache_file=True,
                              cache_file_name=args.preprocess_cache_file,
                              num_proc=args.preprocess_num_process)

    dataset.set_format(type='torch', columns=['input_ids'])
    return dataset


if __name__ == "__main__":
    # Running before 'run.py' to generate a cache for dataset. 
    # Otherwise, each process will generates a cache separately.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments,))
    model_args, data_args = parser.parse_args_into_dataclasses()

    config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir, config=config)
    dataset = get_dataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir)
    dataset.save_to_disk(data_args.preprocess_output_file)
