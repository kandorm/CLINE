import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Union, List, Dict, Tuple
from dataclasses import dataclass
from transformers import DataCollatorForLanguageModeling
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

REPLACE_NONE = -100


@dataclass
class DataCollatorForLEC:
    """
    Data collator used for linguistic error correction task.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for both masked language modeling and linguistic error correction
    """
    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    block_size: int = 512

    def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        input_ids = []
        syn_ant_sent = []
        syn_ant_mask = []
        replace_label = []
        ori_syn_ant_sent = []
        ori_syn_ant_mask = []
        ori_syn_ant_tok_type = []
        ori_syn_ant_label = []

        for example in examples:
            input_ids += [torch.tensor(example["input_ids"], dtype=torch.long)]
            syn_ant_sent += [torch.tensor(example["synonym_antonym_sent"], dtype=torch.long)]
            syn_ant_mask += [torch.tensor([1]*len(example["synonym_antonym_sent"]), dtype=torch.long)]
            replace_label += [torch.tensor(example["replace_label"], dtype=torch.long)]

            # TODO::remove [1:-1]
            syn_id, syn_mask, syn_seg = self.create_features_from_example(example["original_sent"][1:-1], example["synonym_sent"][1:-1]) # drop <s> </s>
            ant_id, ant_mask, ant_seg = self.create_features_from_example(example["original_sent"][1:-1], example["antonym_sent"][1:-1])

            ori_syn_ant_sent += [syn_id, ant_id]
            ori_syn_ant_mask += [syn_mask, ant_mask]
            ori_syn_ant_tok_type += [syn_seg, ant_seg]
            ori_syn_ant_label += [0, 1]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        input_ids, labels = self.mask_tokens(input_ids)

        syn_ant_sent = pad_sequence(syn_ant_sent, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        syn_ant_mask = pad_sequence(syn_ant_mask, batch_first=True, padding_value=0)
        replace_label = pad_sequence(replace_label, batch_first=True, padding_value=REPLACE_NONE)

        ori_syn_ant_sent = pad_sequence(ori_syn_ant_sent, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        ori_syn_ant_mask = pad_sequence(ori_syn_ant_mask, batch_first=True, padding_value=0)
        ori_syn_ant_tok_type = pad_sequence(ori_syn_ant_tok_type, batch_first=True, padding_value=0)
        ori_syn_ant_label = torch.tensor(ori_syn_ant_label, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "synonym_antonym_sent": syn_ant_sent,
            "synonym_antonym_mask": syn_ant_mask,
            "replace_label": replace_label,
            "ori_syn_ant_sent": ori_syn_ant_sent,
            "ori_syn_ant_mask": ori_syn_ant_mask,
            "ori_syn_ant_tok_type": ori_syn_ant_tok_type,
            "ori_syn_ant_label" : ori_syn_ant_label
        }

    def create_features_from_example(self, tokens_a, tokens_b):
        """Creates examples for a single document."""

        max_num_tokens = self.block_size - self.tokenizer.num_special_tokens_to_add(pair=True)

        tokens_a, tokens_b, _ = self.tokenizer.truncate_sequences(
            tokens_a,
            tokens_b,
            num_tokens_to_remove=len(tokens_a) + len(tokens_b) - max_num_tokens,
            truncation_strategy="longest_first",
        )

        input_id = self.tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
        attention_mask = [1] * len(input_id)
        segment_id = self.tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)
        assert len(input_id) <= self.block_size

        input_id = torch.tensor(input_id)
        attention_mask = torch.tensor(attention_mask)
        segment_id = torch.tensor(segment_id)

        return input_id, attention_mask, segment_id

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = REPLACE_NONE  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
