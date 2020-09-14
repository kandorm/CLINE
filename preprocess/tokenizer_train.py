# -*- coding:utf-8 -*-
import os
from argparse import ArgumentParser
from tokenizers import ByteLevelBPETokenizer, CharBPETokenizer
from tokenizers import normalizers

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str, nargs='?', required=True, help="")
    parser.add_argument("--output_path", type=str, nargs='?', required=True, help="")
    
    parser.add_argument("--bytelevel", action="store_true", default=False, help="")
    parser.add_argument("--prefix_space", action="store_true", default=False, help="")
    parser.add_argument("--trim_offsets", action="store_true", default=False, help="")
    parser.add_argument("--lowercase", action="store_true", default=False, help="")
    parser.add_argument("--normalizer", type=str, default="nfkc", nargs='?', help="[nfc, nfd, nfkc, nfkd]")
    parser.add_argument("--bert_normalizer", action="store_true", default=False, help="")
    parser.add_argument("--vocab", type=int, default=52_000, nargs='?', help="")
    parser.add_argument("--minfreq", type=int, default=2, nargs='?', help="")
    args = parser.parse_args()

    file_path = args.input_path
    if os.path.isdir(file_path):
        file_names = os.listdir(file_path)
        file_path = [os.path.join(file_path, fn) for fn in file_names]
    outpath = args.output_path
    if not os.path.exists(outpath):
        os.mkdir(outpath)

    # Initialize a tokenizer
    if args.bytelevel:
        tokenizer = ByteLevelBPETokenizer(add_prefix_space=args.prefix_space,
                                        trim_offsets=args.trim_offsets,
                                        lowercase=args.lowercase,
                                        unicode_normalizer=args.normalizer)
        # tokenizer._tokenizer.normalizer = normalizers.Sequence([
        #     normalizers.Strip(),
        #     normalizers.Lowercase(),
        #     normalizers.NFKC()
        # ])
        # Customize training
        tokenizer.train(files=file_path,
                        vocab_size=args.vocab,
                        min_frequency=args.minfreq,
                        special_tokens=[
                            "<s>",
                            "<pad>",
                            "</s>",
                            "<unk>",
                            "<mask>",
                        ])
    else:
        tokenizer = CharBPETokenizer(suffix="",
                                     lowercase=args.lowercase,
                                     unicode_normalizer=args.normalizer,
                                     bert_normalizer=args.bert_normalizer)
        # Customize training
        tokenizer.train(files=file_path,
                        vocab_size=args.vocab,
                        min_frequency=args.minfreq,
                        suffix="",
                        special_tokens=[
                            "<s>",
                            "<pad>",
                            "</s>",
                            "<unk>",
                            "<mask>",
                        ])
    tokenizer.save_model(outpath)
