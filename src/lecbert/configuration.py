from transformers import RobertaConfig

class LecbertConfig(RobertaConfig):
    num_token_error = 3
