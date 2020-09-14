# -*- coding:utf-8 -*-
import os
from argparse import ArgumentParser
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing, RobertaProcessing

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--token_path", type=str, nargs='?', required=True, help="")
    args = parser.parse_args()

    inputpath = args.token_path
    tokenizer = ByteLevelBPETokenizer(
        os.path.join(inputpath, "vocab.json"),
        os.path.join(inputpath, "merges.txt"),
        add_prefix_space=True,
        trim_offsets=True,
        lowercase=True,
        unicode_normalizer="nfkc"
    )
    tokenizer._tokenizer.post_processor = RobertaProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
        trim_offsets=True,
        add_prefix_space=True
    )
    tokenizer.enable_truncation(max_length=512)
    tokens = tokenizer.encode("I am Julien\nI am from China.").tokens
    print([x.encode('utf-8') for x in tokens])
