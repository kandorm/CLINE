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


import random
import spacy

random.seed(12345)

from spacy.tokens import Doc
Doc.set_extension('_synonym_sent', default=False)
Doc.set_extension('_antonym_sent', default=False)
Doc.set_extension('_syn_ant_sent', default=False)
Doc.set_extension('_replace_intv', default=False)

from wordnet import (
    REPLACE_POS,
    get_synonym,
    get_hypernyms,
    get_antonym,
    get_lemminflect
)

from random_words import RandomWords
rw = RandomWords()

REPLACE_RATIO = 0.15
POSITIVE_RATIO = 0.45
NEGATIVE_RATIO = 0.45
ORIGINAL_RATIO = 0.10

REPLACE_ORIGINAL = 0
REPLACE_LEMMINFLECT = 1
REPLACE_SYNONYM = 2
REPLACE_HYPERNYMS = 3
REPLACE_ADJACENCY = 4
REPLACE_ANTONYM = 5
REPLACE_RANDOM = 6

REPLACE_NONE = -100

SYNONYM_RATIO = 1/3
HYPERNYMS_RATIO = 1/3
LEMMINFLECT_RATIO = 1/3

ANTONYM_RATIO = 1/3
ADJACENCY_RATIO = 1/3
RANDOM_RATIO = 1/3


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
    word_replace: bool = field(
        default=False,
        metadata={"help": "Whether synonym substitution is used to construct adversarial samples."},
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
    lang: Optional[str] = field(
        default="en", metadata={"help": "Language of dataset [en, zh]."}
    )


def get_replace_label(word_list, repl_intv):
    label = [REPLACE_NONE] * len(word_list)
    if not repl_intv:
        return label
    cur_len = 0
    cur_range = 0
    cur_start, cur_end, cur_label = repl_intv[cur_range] # raplacement range is of increasing ordered (include spaces in text)
    for index, word in enumerate(word_list):
        if cur_len >= cur_start and cur_len <= cur_end: # word piece is in replacement range
            label[index] = cur_label

        ## TODO:: it works only in bytelevel-bpe
        cur_len += len(word) # bytelevel contains spaces in the token
        if cur_len > cur_end: # update replacement range
            if cur_range != len(repl_intv)-1: # not the last range
                cur_range += 1
                cur_start, cur_end, cur_label = repl_intv[cur_range]
            else: # no new range
                break
    return label


def get_dataset(
    args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizer,
    evaluate: bool = False,
    cache_dir: Optional[str] = None,
    spacy_nlp=None
):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.load_from_disk:
        return Dataset.load_from_disk(file_path)

    if os.path.isdir(file_path):
        file_names = os.listdir(file_path)
        file_path = [os.path.join(file_path, fn) for fn in file_names]

    dataset = load_dataset("src/text.py", data_files=file_path, split="train", cache_dir=cache_dir, ignore_verifications=True)


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


    def word_replace(examples):
        input_ids = []
        synonym_sent = []
        antonym_sent = []
        synonym_antonym_sent = []
        replace_label = []

        tokenized_text = []
        tokenized_synonym = []
        tokenized_antonym = []
        tokenized_synonym_antonym = []
        tokenized_replace_label = []

        doc_sep_token_id = tokenizer.convert_tokens_to_ids("\n")
        pad_token_id = tokenizer.pad_token_type_id
        block_size = args.block_size - tokenizer.num_special_tokens_to_add(pair=False)

        lines = examples['text']
        docs = spacy_nlp.pipe(lines, n_process=1, batch_size=100, disable=['parser', 'ner'])
        for doc in docs:
            ori_sent = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(" ".join([t.text for t in doc])))
            syn_sent = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(doc._._synonym_sent))
            ant_sent = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(doc._._antonym_sent))

            syn_ant_sent = tokenizer.tokenize(doc._._syn_ant_sent)
            rep_lab = get_replace_label(syn_ant_sent, doc._._replace_intv)
            syn_ant_sent = tokenizer.convert_tokens_to_ids(syn_ant_sent)

            # Synonym substitution will change the token number of BPE, we should pad all sentence to the maximum length
            max_length = max([len(ori_sent), len(syn_sent), len(ant_sent), len(syn_ant_sent)])
            ori_sent += [pad_token_id] * (max_length - len(ori_sent)) + [doc_sep_token_id]
            syn_sent += [pad_token_id] * (max_length - len(syn_sent)) + [doc_sep_token_id]
            ant_sent += [pad_token_id] * (max_length - len(ant_sent)) + [doc_sep_token_id]
            syn_ant_sent += [pad_token_id] * (max_length - len(syn_ant_sent)) + [doc_sep_token_id]
            rep_lab += [REPLACE_NONE] * (max_length - len(rep_lab)) + [REPLACE_NONE]

            tokenized_text.extend(ori_sent)
            tokenized_synonym.extend(syn_sent)
            tokenized_antonym.extend(ant_sent)
            tokenized_synonym_antonym.extend(syn_ant_sent)
            tokenized_replace_label.extend(rep_lab)


        for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
            input_ids.append(
                tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
            )
            synonym_sent.append(
                tokenizer.build_inputs_with_special_tokens(tokenized_synonym[i : i + block_size])
            )
            antonym_sent.append(
                tokenizer.build_inputs_with_special_tokens(tokenized_antonym[i : i + block_size])
            )
            synonym_antonym_sent.append(
                tokenizer.build_inputs_with_special_tokens(tokenized_synonym_antonym[i : i + block_size])
            )
            # TODO:: change to 'build_inputs_with_special_tokens'
            replace_label.append(
                [REPLACE_NONE] + tokenized_replace_label[i : i + block_size] + [REPLACE_NONE]
            )
        # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
        # If your dataset is small, first you should loook for a bigger one :-) and second you
        # can change this behavior by adding (model specific) padding.
        return {'input_ids': input_ids,
                'synonym_sent': synonym_sent,
                'antonym_sent': antonym_sent,
                'synonym_antonym_sent': synonym_antonym_sent,
                'replace_label':replace_label}


    if args.line_by_line:
        dataset = dataset.map(lambda ex: tokenizer(ex["text"], add_special_tokens=True,
                                                   truncation=True, max_length=args.block_size),
                              batched=True,
                              batch_size=args.preprocess_batch_size,
                              writer_batch_size=args.preprocess_batch_size,
                              remove_columns=dataset.column_names,
                              load_from_cache_file=True,
                              cache_file_name=args.preprocess_cache_file,
                              num_proc=args.preprocess_num_process)
        dataset.set_format(type='torch', columns=['input_ids'])

    elif args.word_replace:
        dataset = dataset.map(word_replace,
                              batched=True,
                              batch_size=args.preprocess_batch_size,
                              writer_batch_size=args.preprocess_batch_size,
                              remove_columns=dataset.column_names,
                              load_from_cache_file=True,
                              cache_file_name=args.preprocess_cache_file,
                              num_proc=args.preprocess_num_process)
        dataset.set_format(type='torch', columns=['input_ids', 'synonym_sent', 'antonym_sent', 'synonym_antonym_sent', 'replace_label'])

    else:
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


def replace_word(doc):
    synonym_sent = []
    antonym_sent = []
    syn_ant_sent = []
    replace_intv = []

    length = len(doc)
    rep_num = int(length*REPLACE_RATIO)

    rep_index = []
    pos_word = {p:[] for p in REPLACE_POS}
    cur_pos = {p:0 for p in REPLACE_POS}
    for index, token in enumerate(doc):
        if token.pos_ in REPLACE_POS:
            rep_index.append(index)
            pos_word[token.pos_].append(token.text)
    rep_num = min(rep_num, len(rep_index))
    rep_index = random.sample(rep_index, rep_num)

    cur_len = -1 # point to the space before next token
    for index, token in enumerate(doc):
        syn = ant = mx = token.text

        if index in rep_index:
            rep_type = REPLACE_ORIGINAL

            syn_ant_rand = random.random()
            if syn_ant_rand < POSITIVE_RATIO:

                syn_rand = random.random()

                if syn_rand < SYNONYM_RATIO: # synonym replacement
                    syns = get_synonym(token)
                    syn = random.choice(syns) if syns else token.text
                    if syn != token.text:
                        rep_type = REPLACE_SYNONYM

                if rep_type == REPLACE_ORIGINAL and syn_rand < SYNONYM_RATIO + HYPERNYMS_RATIO: # hypernyms replacement
                    syns = get_hypernyms(token)
                    syn = random.choice(syns) if syns else token.text
                    if syn != token.text:
                        rep_type = REPLACE_HYPERNYMS

                if rep_type == REPLACE_ORIGINAL: # lemminflect
                    syns = get_lemminflect(token)
                    syn = random.choice(syns) if syns else token.text
                    if syn != token.text:
                        rep_type = REPLACE_LEMMINFLECT

                mx = syn

            elif syn_ant_rand < POSITIVE_RATIO + NEGATIVE_RATIO:

                ant_rand = random.random()

                if ant_rand < ANTONYM_RATIO: # antonym replacement
                    ants = get_antonym(token)
                    ant = random.choice(ants) if ants else token.text
                    if ant != token.text:
                        rep_type = REPLACE_ANTONYM

                if rep_type == REPLACE_ORIGINAL and ant_rand < ANTONYM_RATIO + ADJACENCY_RATIO:
                    c_p = cur_pos[token.pos_]
                    s_p = min(0, c_p-2)
                    ants = pos_word[token.pos_][s_p:c_p] + pos_word[token.pos_][c_p+1:c_p+3]
                    ant = random.choice(ants) if ants else token.text
                    if ant != token.text:
                        rep_type = REPLACE_ADJACENCY

                if rep_type == REPLACE_ORIGINAL: # hypernyms replacement
                    ant = rw.random_word()
                    if ant != token.text:
                        rep_type = REPLACE_RANDOM

                mx = ant

            replace_intv.append((cur_len, cur_len + len(mx.encode('utf-8')), rep_type)) # fix length mismatch, mx.encode for bytelevelbpe

        if token.pos_ in REPLACE_POS:
            cur_pos[token.pos_] += 1

        cur_len = cur_len + len(mx.encode('utf-8')) + 1 # point to the space before next token
        synonym_sent.append(syn)
        antonym_sent.append(ant)
        syn_ant_sent.append(mx)


    doc._._synonym_sent = " ".join(synonym_sent)
    doc._._antonym_sent = " ".join(antonym_sent)
    doc._._syn_ant_sent = " ".join(syn_ant_sent)
    doc._._replace_intv = replace_intv

    return doc


if __name__ == "__main__":
    # Running before 'run.py' to generate a cache for dataset. 
    # Otherwise, each process will generates a cache separately.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments,))
    model_args, data_args = parser.parse_args_into_dataclasses()

    spacy_nlp = spacy.load(data_args.lang)
    spacy_nlp.add_pipe(replace_word, last=True)

    config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir, config=config)
    dataset = get_dataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir, spacy_nlp=spacy_nlp)
    dataset.save_to_disk(data_args.preprocess_output_file)

    # txt = ["Blue Network The Blue Network (previously the NBC Blue Network) was the on-air name of the now defunct American radio network, which ran from 1927 to 1945."]
    # docs = spacy_nlp.pipe(txt, n_process=1, batch_size=100, disable=['parser'])
    # for doc in docs:
    #     ori_sent = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(" ".join([t.text for t in doc])))
    #     syn_sent = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(doc._._synonym_sent))
    #     ant_sent = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(doc._._antonym_sent))

    #     syn_ant_sent = tokenizer.tokenize(doc._._syn_ant_sent)
    #     rep_lab = get_replace_label(syn_ant_sent, doc._._replace_intv)
    #     syn_ant_id = tokenizer.convert_tokens_to_ids(syn_ant_sent)

    #     print(" ".join([t.text for t in doc]))
    #     print(doc._._synonym_sent)
    #     print(doc._._antonym_sent)
    #     print(doc._._syn_ant_sent)
    #     print(doc._._replace_intv)

    #     print(ori_sent)
    #     print([(syn_ant_sent[i], rep_lab[i]) for i in range(len(syn_ant_sent))])
    #     tokens = tokenizer.build_inputs_with_special_tokens(syn_ant_id)
    #     print(tokens)
