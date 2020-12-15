import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Tuple
from dataclasses import dataclass
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
        batch_size = len(examples)
        block_size = self.block_size - self.tokenizer.num_special_tokens_to_add(pair=False)

        ori_sent = []
        ori_mask = []
        syn_sent = []
        syn_mask = []
        ant_sent = []
        ant_mask = []
        ori_label = []
        syn_label = []
        ant_label = []

        for example in examples:
            ori_sen = self.tokenizer.build_inputs_with_special_tokens(example["input_ids"][:block_size])
            ori_lab = self.tokenizer.create_token_label_from_sequences([REPLACE_NONE]*len(example["input_ids"][:block_size]))
            syn_sen = self.tokenizer.build_inputs_with_special_tokens(example["synonym_ids"][:block_size])
            syn_lab = example["synonym_label"][:block_size]
            syn_lab = [1 if lb not in [REPLACE_NONE, 0] else lb for lb in syn_lab]
            syn_lab = self.tokenizer.create_token_label_from_sequences(syn_lab)
            ant_sen = self.tokenizer.build_inputs_with_special_tokens(example["antonym_ids"][:block_size])
            ant_lab = example["antonym_label"][:block_size]
            ant_lab = [2 if lb not in [REPLACE_NONE, 0] else lb for lb in ant_lab]
            ant_lab = self.tokenizer.create_token_label_from_sequences(ant_lab)

            ori_sent += [torch.tensor(ori_sen, dtype=torch.long)]
            ori_mask += [torch.ones(len(ori_sen))]
            syn_sent += [torch.tensor(syn_sen, dtype=torch.long)]
            syn_mask += [torch.ones(len(syn_sen))]
            ant_sent += [torch.tensor(ant_sen, dtype=torch.long)]
            ant_mask += [torch.ones(len(ant_sen))]

            ori_label += [torch.tensor(ori_lab, dtype=torch.long)]
            syn_label += [torch.tensor(syn_lab, dtype=torch.long)]
            ant_label += [torch.tensor(ant_lab, dtype=torch.long)]

        input_ids = ori_sent + syn_sent + ant_sent
        attention_mask = ori_mask + syn_mask + ant_mask
        labels = ori_label + syn_label + ant_label

        assert len(input_ids) == batch_size * 3
        assert len(attention_mask) == batch_size * 3
        assert len(labels) == batch_size * 3

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=REPLACE_NONE)

        mlm_sent, mlm_label = self.mask_tokens(input_ids[:batch_size])
        input_ids[:batch_size] = mlm_sent
        labels[:batch_size] = mlm_label

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


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
