import warnings

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss
from dataclasses import dataclass

from typing import Optional, Tuple

from transformers.activations import ACT2FN, gelu
from transformers.file_utils import ModelOutput
from transformers.modeling_roberta import RobertaModel, RobertaPreTrainedModel


# Copied from transformers.modeling_roberta.RobertaLMHead
class LecbertLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x


# Copied from transformers.modeling_roberta.RobertaLMHead with config.vocab_size->config.num_token_error
class LecbertTECHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.num_token_error, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.num_token_error))

        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of labels
        x = self.decoder(x)

        return x


class LecbertForPreTraining(RobertaPreTrainedModel):
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.mlm_head = LecbertLMHead(config)
        self.tokn_classifier = LecbertTECHead(config)
        self.log_vars = nn.Parameter(torch.zeros(3))

        self.init_weights()

    def get_output_embeddings(self):
        return self.mlm_head.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        antonym_ids=None,
        antonym_label=None,
        synonym_ids=None,
        synonym_label=None,
        **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        replace_label (``torch.LongTensor`` of shape ``(batch_size,sequence_length)``, `optional`):
            Labels for computing the token replace type prediction (classification) loss.
            Indices should be in ``[0, 1, 2, 3, 4, 5, 6]``:
            - 0 indicates the token is the original token,
            - 1 indicates the token is replaced with the lemminflect token,
            - 2 indicates the token is replaced with the synonym,
            - 3 indicates the token is replaced with the hypernyms,
            - 4 indicates the token is replaced with the adjacency,
            - 5 indicates the token is replaced with the antonym,
            - 6 indicates the token is replaced with the random word.
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        Returns:
        """
        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("masked_lm_labels")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Masked Language Model
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]

        batch_size = input_ids.size(0) // 3
        ori_seq, syn_ant_seq = sequence_output[:batch_size], sequence_output[batch_size:]
        mlm_labels, tec_labels = labels[:batch_size], labels[batch_size:]
        mlm_scores = self.mlm_head(ori_seq)
        tec_scores = self.tokn_classifier(syn_ant_seq)

        ori_sen, syn_sen, ant_sen = pooled_output[:batch_size], pooled_output[batch_size:batch_size*2], pooled_output[batch_size*2:]
        ori_syn_rel = torch.sigmoid(torch.mean(ori_sen * syn_sen, dim=-1, keepdim=True))
        ori_ant_rel = torch.sigmoid(torch.mean(ori_sen * ant_sen, dim=-1, keepdim=True))
        sec_scores = torch.cat((ori_syn_rel, ori_ant_rel), dim=0)
        sec_labels = torch.cat((torch.ones(batch_size), torch.zeros(batch_size)), dim=0).to(labels.device)

        total_loss = None
        if labels is not None:
            loss_tok = CrossEntropyLoss()
            mlm_loss = loss_tok(mlm_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))
            tec_loss = loss_tok(tec_scores.view(-1, self.config.num_token_error), tec_labels.view(-1))

            loss_sen = BCELoss()
            sec_loss = loss_sen(sec_scores.view(-1), sec_labels.view(-1))

            # total_loss = mlm_loss + tec_loss + sec_loss
            total_loss = torch.exp(-self.log_vars[0]) * mlm_loss + torch.clamp(self.log_vars[0], min=0) + \
                         torch.exp(-self.log_vars[1]) * tec_loss + torch.clamp(self.log_vars[1], min=0) + \
                         torch.exp(-self.log_vars[2]) * sec_loss + torch.clamp(self.log_vars[2], min=0)

            #print(mlm_loss.item(), tec_loss.item(), sec_loss.item())

        if not return_dict:
            output = (mlm_scores,) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return LecbertForPretrainingOutput(
            loss=total_loss,
            prediction_logits=mlm_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@dataclass
class LecbertForPretrainingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
