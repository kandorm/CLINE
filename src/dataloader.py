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
POSITIVE_RATIO = 1/3
NEGATIVE_RATIO = 1/3
ORIGINAL_RATIO = 1/3

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
        original_sent = []
        synonym_sent = []
        antonym_sent = []
        synonym_antonym_sent = []
        replace_label = []

        block_size = args.block_size - 2
        bos_token_id = tokenizer.bos_token_id
        eos_token_id = tokenizer.eos_token_id

        lines = examples['text']
        docs = spacy_nlp.pipe(lines, n_process=1, batch_size=100, disable=['parser', 'ner'])
        for doc in docs:
            if len(doc) < 2: continue  # line break
            ori_sent = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(" ".join([t.text for t in doc])))
            syn_sent = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(doc._._synonym_sent))
            ant_sent = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(doc._._antonym_sent))

            syn_ant_sent = tokenizer.tokenize(doc._._syn_ant_sent)
            rep_lab = get_replace_label(syn_ant_sent, doc._._replace_intv)
            syn_ant_sent = tokenizer.convert_tokens_to_ids(syn_ant_sent)

            ori_sent = [bos_token_id] + ori_sent[:block_size] + [eos_token_id]
            syn_sent = [bos_token_id] + syn_sent[:block_size] + [eos_token_id]
            ant_sent = [bos_token_id] + ant_sent[:block_size] + [eos_token_id]
            syn_ant_sent = [bos_token_id] + syn_ant_sent[:block_size] + [eos_token_id]
            rep_lab = [REPLACE_NONE] + rep_lab[:block_size] + [REPLACE_NONE]

            original_sent.append(ori_sent)
            synonym_sent.append(syn_sent)
            antonym_sent.append(ant_sent)
            synonym_antonym_sent.append(syn_ant_sent)
            replace_label.append(rep_lab)

        return {'original_sent': original_sent,
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
        dataset.set_format(type=None, columns=['input_ids'])

    elif args.word_replace:
        dataset = dataset.map(word_replace,
                              batched=True,
                              batch_size=args.preprocess_batch_size,
                              writer_batch_size=args.preprocess_batch_size,
                              remove_columns=dataset.column_names,
                              load_from_cache_file=True,
                              cache_file_name=args.preprocess_cache_file,
                              num_proc=args.preprocess_num_process)
        dataset.set_format(type=None, columns=['original_sent', 'synonym_sent', 'antonym_sent', 'synonym_antonym_sent', 'replace_label'])

    else:
        dataset = dataset.map(lines_to_block,
                              batched=True,
                              batch_size=args.preprocess_batch_size,
                              writer_batch_size=args.preprocess_batch_size,
                              remove_columns=dataset.column_names,
                              load_from_cache_file=True,
                              cache_file_name=args.preprocess_cache_file,
                              num_proc=args.preprocess_num_process)
        dataset.set_format(type=None, columns=['input_ids'])

    return dataset


def search_replacement(doc, candidate_index, replace_type, max_num, pos_to_words=None):
    sr_rep = []
    if max_num < 1:
        return sr_rep

    for r_idx in candidate_index:
        token = doc[r_idx]
        rep = None
        if replace_type == REPLACE_ANTONYM:
            reps = get_antonym(token)
            rep = random.choice(reps) if reps else None
        elif replace_type == REPLACE_ADJACENCY:
            reps = pos_to_words[token.pos_]
            rep = random.choice(reps) if reps else None
        elif replace_type == REPLACE_RANDOM:
            rep = rw.random_word()
        elif replace_type == REPLACE_SYNONYM:
            reps = get_synonym(token)
            rep = random.choice(reps) if reps else None
        elif replace_type == REPLACE_HYPERNYMS:
            reps = get_hypernyms(token)
            rep = random.choice(reps) if reps else None
        elif replace_type == REPLACE_LEMMINFLECT:
            reps = get_lemminflect(token)
            rep = random.choice(reps) if reps else None
        else:
            pass

        if rep and rep.lower() != token.text.lower():
            sr_rep.append((r_idx, rep, replace_type))

        if len(sr_rep) >= max_num:
            break

    return sr_rep


def replace_word(doc):
    synonym_sent = []
    antonym_sent = []
    syn_ant_sent = []
    replace_intv = []

    length = len(doc)
    rep_num = int(length*REPLACE_RATIO)

    rep_index = []
    pos_word = {p:[] for p in REPLACE_POS}
    for index, token in enumerate(doc):
        if token.pos_ in REPLACE_POS:
            rep_index.append(index)
            pos_word[token.pos_].append(token.text)

    rep_num = min(rep_num, len(rep_index))
    ori_num = int(rep_num * ORIGINAL_RATIO)
    syn_num = int(rep_num * (ORIGINAL_RATIO+POSITIVE_RATIO)) - ori_num
    ant_num = rep_num - ori_num - syn_num

    syn_rand = random.random()
    ant_rand = random.random()
    cand_index = rep_index[:]
    random.shuffle(cand_index)

    syn_replace = []
    ant_replace = [] # [(rep_idx, rep_word, rep_type)]
    ori_replace = []

    ############### Antonym Replacement ####################
    if ant_rand < ANTONYM_RATIO:
        ant_replace = search_replacement(doc, candidate_index=cand_index, replace_type=REPLACE_ANTONYM, max_num=ant_num)

    if not ant_replace and ant_rand < ANTONYM_RATIO + ADJACENCY_RATIO:
        ant_replace = search_replacement(doc, candidate_index=cand_index, replace_type=REPLACE_ADJACENCY, max_num=ant_num, pos_to_words=pos_word)

    if not ant_replace:
        ant_replace = search_replacement(doc, candidate_index=cand_index, replace_type=REPLACE_RANDOM, max_num=ant_num)

    for r in ant_replace:
        ant_rep_idx = r[0]
        cand_index.remove(ant_rep_idx)

    ############### Synonym Replacement ####################
    if syn_rand < HYPERNYMS_RATIO:
        syn_replace = search_replacement(doc, candidate_index=cand_index, replace_type=REPLACE_HYPERNYMS, max_num=syn_num)

    if not syn_replace and syn_rand < HYPERNYMS_RATIO + SYNONYM_RATIO:
        syn_replace = search_replacement(doc, candidate_index=cand_index, replace_type=REPLACE_SYNONYM, max_num=syn_num)

    if not syn_replace:
        syn_replace = search_replacement(doc, candidate_index=cand_index, replace_type=REPLACE_LEMMINFLECT, max_num=syn_num)

    for r in syn_replace:
        syn_rep_idx = r[0]
        cand_index.remove(syn_rep_idx)

    ############### Original Replacement ####################
    for r_idx in random.sample(cand_index, ori_num):
        ori_replace.append((r_idx, doc[r_idx].text, REPLACE_ORIGINAL))

    all_replace = ant_replace + syn_replace + ori_replace
    all_replace = sorted(all_replace, key=lambda x:x[0], reverse=True)

    cur_len = -1 # point to the space before next token
    rep_idx, rep_word, rep_type = all_replace.pop() if all_replace else (None, None, None)
    for index, token in enumerate(doc):
        syn = ant = mx = token.text

        if index == rep_idx:
            if rep_type in [REPLACE_SYNONYM, REPLACE_HYPERNYMS, REPLACE_LEMMINFLECT]:
                syn = rep_word
                mx = syn
            elif rep_type in [REPLACE_ANTONYM, REPLACE_ADJACENCY, REPLACE_RANDOM]:
                ant = rep_word
                mx = ant
            else:
                mx = rep_word

            replace_intv.append((cur_len, cur_len + len(mx.encode('utf-8')), rep_type)) # fix length mismatch, mx.encode for bytelevelbpe
            rep_idx, rep_word, rep_type = all_replace.pop() if all_replace else (None, None, None)

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

    spacy_nlp = spacy.load(data_args.lang) # 'en_core_web_sm'
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
