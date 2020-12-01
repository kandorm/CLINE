from transformers import RobertaTokenizer
from typing import List, Optional

REPLACE_NONE = -100


class LecbertTokenizer(RobertaTokenizer):
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A RoBERTa sequence has the following format:

        - single sequence: ``<s> X </s>``
        - pair of sequences: ``<s> A </s></s> B </s>``

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def create_token_label_from_sequences(
        self, labels_0: List[int], labels_1: Optional[List[int]] = None
    ) -> List[int]:

        cls = [REPLACE_NONE]
        sep = [REPLACE_NONE]
        if labels_1 is None:
            return cls + labels_0 + sep
        return cls + labels_0 + sep + sep + labels_1 + sep
