from pathlib import Path
from typing import Optional

import torch
from transformers import AutoConfig, AutoModel
from transformers.modeling_utils import SequenceSummary

from thebatnews.modeling.mask import ILanguageModel


class IDIBERT(ILanguageModel):
    """
    A DistilBERT model that wraps HuggingFace's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.

    NOTE:
    - DistilBert doesn’t have token_type_ids, you don’t need to indicate which
    token belongs to which segment. Just separate your segments with the separation
    token tokenizer.sep_token (or [SEP])
    - Unlike the other BERT variants, DistilBert does not output the
    pooled_output. An additional pooler is initialized.

    """

    def __init__(self):
        self.name = "IDIBERT"
        super(IDIBERT, self).__init__()
        self.model = None
        self.pooler = None

    @classmethod
    def load(cls, pretrained_model_name_or_path, language=None, **kwargs):
        """
        Load a pretrained model by supplying

        * the name of a remote model on s3 ("distilbert-base-german-cased" ...)
        * OR a local path of a model trained via transformers ("some_dir/huggingface_model")
        * OR a local path of a model trained via FARM ("some_dir/farm_model")

        :param pretrained_model_name_or_path: The path of the saved pretrained model or its name.
        :type pretrained_model_name_or_path: str

        """

        _model = cls()
        if "farm_lm_name" in kwargs:
            _model.name = kwargs["farm_lm_name"]
        else:
            _model.name = pretrained_model_name_or_path
        # We need to differentiate between loading model using FARM format and Pytorch-Transformers format
        farm_lm_config = Path(pretrained_model_name_or_path) / "language_model_config.json"
        if Path(farm_lm_config).exists():
            # FARM style
            config = AutoConfig.from_pretrained(farm_lm_config)
            farm_lm_model = Path(pretrained_model_name_or_path) / "language_model.bin"
            _model.model = AutoModel.from_pretrained(farm_lm_model, config=config, **kwargs)
            _model.language = _model.model.config.language
        else:
            # Pytorch-transformer Style
            _model.model = AutoModel.from_pretrained(str(pretrained_model_name_or_path), **kwargs)
            _model.language = "multilingual"
        config = _model.model.config

        # DistilBERT does not provide a pooled_output by default. Therefore, we need to initialize an extra pooler.
        # The pooler takes the first hidden representation & feeds it to a dense layer of (hidden_dim x hidden_dim).
        # We don't want a dropout in the end of the pooler, since we do that already in the adaptive model before we
        # feed everything to the prediction head
        config.summary_last_dropout = 0
        config.summary_type = "first"
        config.summary_activation = "tanh"
        _model.pooler = SequenceSummary(config)
        _model.pooler.apply(_model.model._init_weights)
        return _model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        segment_ids: Optional[torch.Tensor],  # DistilBERT does not use them, see DistilBERTLanguageModel
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: bool = False,
    ):
        """
        Perform the forward pass of the DistilBERT model.

        :param input_ids: The ids of each token in the input sequence. Is a tensor of shape [batch_size, max_seq_len]
        :type input_ids: torch.Tensor
        :param padding_mask: A mask that assigns a 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len]
        :return: Embeddings for each token in the input sequence.

        """
        output_tuple = self.model(
            input_ids,
            attention_mask=attention_mask,
        )
        # We need to manually aggregate that to get a pooled output (one vec per seq)
        pooled_output = self.pooler(output_tuple[0])
        if self.model.config.output_hidden_states is True:
            sequence_output, _ = output_tuple[0], output_tuple[1]
            return sequence_output, pooled_output
        else:
            return None, pooled_output


class IE5Model(ILanguageModel):
    """
    See https://huggingface.co/intfloat/multilingual-e5-base for model details
    """
    def __init__(self):
        super(IE5Model, self).__init__()
        self.model = None
        self.name = "IE5Model"

    def average_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    @classmethod
    def load(cls, pretrained_model_name_or_path, language=None, **kwargs):
        """
        Load a pretrained model by supplying

        * the name of a remote model on s3 ("distilbert-base-german-cased" ...)
        * OR a local path of a model trained via transformers ("some_dir/huggingface_model")
        * OR a local path of a model trained via FARM ("some_dir/farm_model")

        :param pretrained_model_name_or_path: The path of the saved pretrained model or its name.
        :type pretrained_model_name_or_path: str

        """

        _model = cls()
        if "farm_lm_name" in kwargs:
            _model.name = kwargs["farm_lm_name"]
        else:
            _model.name = pretrained_model_name_or_path
        # We need to differentiate between loading model using FARM format and Pytorch-Transformers format
        farm_lm_config = Path(pretrained_model_name_or_path) / "language_model_config.json"
        if Path(farm_lm_config).exists():
            # FARM style
            config = AutoConfig.from_pretrained(farm_lm_config)
            farm_lm_model = Path(pretrained_model_name_or_path) / "language_model.bin"
            _model.model = AutoModel.from_pretrained(farm_lm_model, config=config, **kwargs)
            _model.language = _model.model.config.language
        else:
            # Pytorch-transformer Style
            _model.model = AutoModel.from_pretrained(str(pretrained_model_name_or_path), **kwargs)
            _model.language = "multilingual"
        return _model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        segment_ids: Optional[torch.Tensor],
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: bool = False,
    ):
        """
        Perform the forward pass of the intfloat/e5 model.

        :param input_ids: The ids of each token in the input sequence. Is a tensor of shape [batch_size, max_seq_len]
        :type input_ids: torch.Tensor
        :param padding_mask: A mask that assigns a 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len]
        :param segment_ids: A mask that differentiate between sentences.
        :return: Embeddings for each token in the input sequence.

        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.average_pool(outputs.last_hidden_state, attention_mask) # batch_size x max_seq_len x hidden_size
        return None, embeddings


__all__ = ["IDIBERT", "IE5Model"]
