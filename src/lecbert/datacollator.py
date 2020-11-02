import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Union, List, Dict
from dataclasses import dataclass
from transformers import DataCollatorForLanguageModeling


REPLACE_NONE = -100


@dataclass
class DataCollatorForLEC(DataCollatorForLanguageModeling):
    """
    Data collator used for linguistic error correction task.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for both masked language modeling and linguistic error correction
    """

    def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        input_ids = []
        original_sent = []
        original_mask = []
        synonym_sent = []
        synonym_mask = []
        antonym_sent = []
        antonym_mask = []
        syn_ant_sent = []
        syn_ant_mask = []
        replace_label = []

        for example in examples:
            input_ids.append(example["input_ids"])
            original_sent.append(torch.tensor(example["original_sent"], dtype=torch.long))
            original_mask.append(torch.tensor([1]*len(example["original_sent"])))
            synonym_sent.append(torch.tensor(example["synonym_sent"], dtype=torch.long))
            synonym_mask.append(torch.tensor([1]*len(example["synonym_sent"])))
            antonym_sent.append(torch.tensor(example["antonym_sent"], dtype=torch.long))
            antonym_mask.append(torch.tensor([1]*len(example["antonym_sent"])))
            syn_ant_sent.append(torch.tensor(example["synonym_antonym_sent"], dtype=torch.long))
            syn_ant_mask.append(torch.tensor([1]*len(example["synonym_antonym_sent"])))
            replace_label.append(torch.tensor(example["replace_label"], dtype=torch.long))

        input_ids = self._tensorize_batch(input_ids)
        input_ids, labels = self.mask_tokens(input_ids)

        ori_syn_ant_sent = original_sent + synonym_sent + antonym_sent
        ori_syn_ant_sent = pad_sequence(ori_syn_ant_sent, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        ori_syn_ant_mask = original_mask + synonym_mask + antonym_mask
        ori_syn_ant_mask = pad_sequence(ori_syn_ant_mask, batch_first=True, padding_value=0)
        syn_ant_sent = pad_sequence(syn_ant_sent, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        syn_ant_mask = pad_sequence(syn_ant_mask, batch_first=True, padding_value=0)
        replace_label = pad_sequence(replace_label, batch_first=True, padding_value=REPLACE_NONE)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "ori_syn_ant_sent": ori_syn_ant_sent,
            "ori_syn_ant_mask": ori_syn_ant_mask,
            "synonym_antonym_sent": syn_ant_sent,
            "synonym_antonym_mask": syn_ant_mask,
            "replace_label": replace_label,
        }
