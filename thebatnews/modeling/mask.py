import abc
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import simplejson as json
import torch
import torch.nn as nn

from thebatnews.etc.error import ModelingError
from thebatnews.etc.format import is_json
from thebatnews.modeling import div

#: Names of the attributes in various model configs which refer to the number of dimensions in the output vectors
OUTPUT_DIM_NAMES = ["dim", "hidden_size", "d_model"]

logger = logging.getLogger(__name__)


class IFlow(nn.Module):
    """
    Takes word embeddings from a language model and generates logits for a given task. Can also convert logits
    to loss and and logits to predictions.
    """

    subclasses = {}  # type: Dict

    def __init_subclass__(cls, **kwargs):
        """This automatically keeps track of all available subclasses.
        Enables generic load() for all specific PredictionHead implementation.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    @classmethod
    def create(cls, prediction_head_name: str, layer_dims: List[int], class_weights=Optional[List[float]]):
        """
        Create subclass of Prediction Head.

        :param prediction_head_name: Classname (exact string!) of prediction head we want to create
        :param layer_dims: describing the feed forward block structure, e.g. [768,2]
        :param class_weights: The loss weighting to be assigned to certain label classes during training.
           Used to correct cases where there is a strong class imbalance.
        :return: Prediction Head of class prediction_head_name
        """
        # TODO make we want to make this more generic.
        #  1. Class weights is not relevant for all heads.
        #  2. Layer weights impose FF structure, maybe we want sth else later
        # Solution: We could again use **kwargs
        return cls.subclasses[prediction_head_name](layer_dims=layer_dims, class_weights=class_weights)

    def save_config(self, save_dir: Union[str, Path], head_num: int = 0):
        """
        Saves the config as a json file.

        :param save_dir: Path to save config to
        :param head_num: Which head to save
        """
        # updating config in case the parameters have been changed
        self.generate_config()
        output_config_file = Path(save_dir) / f"prediction_head_{head_num}_config.json"
        with open(output_config_file, "w") as file:
            json.dump(self.config, file)

    def save(self, save_dir: Union[str, Path], head_num: int = 0):
        """
        Saves the prediction head state dict.

        :param save_dir: path to save prediction head to
        :param head_num: which head to save
        """
        output_model_file = Path(save_dir) / f"prediction_head_{head_num}.bin"
        torch.save(self.state_dict(), output_model_file)
        self.save_config(save_dir, head_num)

    def generate_config(self):
        """
        Generates config file from Class parameters (only for sensible config parameters).
        """
        config = {}
        for key, value in self.__dict__.items():
            if type(value) is np.ndarray:
                value = value.tolist()
            if is_json(value) and key[0] != "_":
                config[key] = value
            if self.task_name == "text_similarity" and key == "similarity_function":
                config["similarity_function"] = value
        config["name"] = self.__class__.__name__
        config.pop("config", None)
        self.config = config

    @classmethod
    def load(cls, config_file: str, strict: bool = True, load_weights: bool = True):
        """
        Loads a Prediction Head. Infers the class of prediction head from config_file.

        :param config_file: location where corresponding config is stored
        :param strict: whether to strictly enforce that the keys loaded from saved model match the ones in
                       the PredictionHead (see torch.nn.module.load_state_dict()).
        :param load_weights: whether to load weights of the prediction head
        :return: PredictionHead
        :rtype: PredictionHead[T]
        """
        with open(config_file) as f:
            config = json.load(f)
        prediction_head = cls.subclasses[config["name"]](**config)
        if load_weights:
            model_file = cls._get_model_file(config_file=config_file)
            logger.info("Loading prediction head from %s", model_file)
            prediction_head.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")), strict=strict)
        return prediction_head

    def logits_to_loss(self, logits, labels):
        """
        Implement this function in your special Prediction Head.
        Should combine logits and labels with a loss fct to a per sample loss.

        :param logits: logits, can vary in shape and type, depending on task
        :param labels: labels, can vary in shape and type, depending on task
        :return: per sample loss as a torch.tensor of shape [batch_size]
        """
        raise NotImplementedError()

    def logits_to_preds(self, logits, span_mask, start_of_word, seq_2_start_t, max_answer_length, **kwargs):
        """
        Implement this function in your special Prediction Head.
        Should combine turn logits into predictions.

        :param logits: logits, can vary in shape and type, depending on task
        :return: predictions as a torch.tensor of shape [batch_size]
        """
        raise NotImplementedError()

    def prepare_labels(self, **kwargs):
        """
        Some prediction heads need additional label conversion.

        :param kwargs: placeholder for passing generic parameters
        :return: labels in the right format
        :rtype: object
        """
        # TODO maybe just return **kwargs to not force people to implement this
        raise NotImplementedError()

    def resize_input(self, input_dim):
        """
        This function compares the output dimensionality of the language model against the input dimensionality
        of the prediction head. If there is a mismatch, the prediction head will be resized to fit.
        """
        # Note on pylint disable
        # self.feed_forward's existence seems to be a condition for its own initialization
        # within this class, which is clearly wrong. The only way this code could ever be called is
        # thanks to subclasses initializing self.feed_forward somewhere else; however, this is a
        # very implicit requirement for subclasses, and in general bad design. FIXME when possible.
        if "feed_forward" not in dir(self):
            return
        else:
            old_dims = self.feed_forward.layer_dims  # pylint: disable=access-member-before-definition
            if input_dim == old_dims[0]:
                return
            new_dims = [input_dim] + old_dims[1:]
            logger.info(
                "Resizing input dimensions of %s (%s) from %s to %s to match language model",
                type(self).__name__,
                self.task_name,
                old_dims,
                new_dims,
            )
            self.feed_forward = div.FeedForwardBlock(new_dims)
            self.layer_dims[0] = input_dim
            self.feed_forward.layer_dims[0] = input_dim

    @classmethod
    def _get_model_file(cls, config_file: Union[str, Path]):
        if "config.json" in str(config_file) and "prediction_head" in str(config_file):
            head_num = int("".join([char for char in os.path.basename(config_file) if char.isdigit()]))
            model_file = Path(os.path.dirname(config_file)) / f"prediction_head_{head_num}.bin"
        else:
            raise ValueError(f"This doesn't seem to be a proper prediction_head config file: '{config_file}'")
        return model_file

    def _set_name(self, name):
        self.task_name = name


# TODO analyse if LMs can be completely used through HF transformers
class ILanguageModel(nn.Module, abc.ABC):
    """
    The parent class for any kind of model that can embed language into a semantic vector space.
    These models read in tokenized sentences and return vectors that capture the meaning of sentences or of tokens.
    """

    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        """This automatically keeps track of all available subclasses.
        Enables generic load() or all specific LanguageModel implementation.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def __init__(self):
        super().__init__()
        self._output_dims = None

    @classmethod
    def load(cls, pretrained_model_name_or_path, revision=None, n_added_tokens=0, language_model_class=None, **kwargs):
        config_file = Path(pretrained_model_name_or_path) / "language_model_config.json"
        assert config_file.exists(), "The config is not found, couldn't load the model"
        logger.info(f"Model found locally at {pretrained_model_name_or_path}")
        # it's a local directory in FARM format
        with open(config_file) as f:
            config = json.load(f)
        language_model = cls.subclasses[config["klass"]].load(pretrained_model_name_or_path)
        return language_model

    @property
    def encoder(self):
        return self.model.encoder

    @abc.abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        segment_ids: Optional[torch.Tensor],  # DistilBERT does not use them, see DistilBERTLanguageModel
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: bool = False,
    ):
        raise NotImplementedError

    @property
    def output_hidden_states(self):
        """
        Controls whether the model outputs the hidden states or not
        """
        self.encoder.config.output_hidden_states = True

    @output_hidden_states.setter
    def output_hidden_states(self, value: bool):
        """
        Sets the model to output the hidden states or not
        """
        self.encoder.config.output_hidden_states = value

    @property
    def output_dims(self):
        """
        The output dimension of this language model
        """
        if self._output_dims:
            return self._output_dims

        for odn in OUTPUT_DIM_NAMES:
            try:
                value = getattr(self.model.config, odn, None)
                if value:
                    self._output_dims = value
                    return value
            except AttributeError:
                raise ModelingError("Can't get the output dimension before loading the model.")

        raise ModelingError("Could not infer the output dimensions of the language model.")

    def save_config(self, save_dir: Union[Path, str]):
        """
        Save the configuration of the language model in format.
        """
        save_filename = Path(save_dir) / "language_model_config.json"
        setattr(self.model.config, "name", self.name)  # type: ignore [union-attr]
        setattr(self.model.config, "language", self.language)  # type: ignore [union-attr]
        config = self.model.config.to_dict()
        config["klass"] = self.__class__.__name__
        # string = json.sto_json_string()  # type: ignore [union-attr,operator]
        with open(str(save_filename), "w") as f:
            f.write(json.dumps(config))

    def save(self, save_dir: Union[str, Path], state_dict: Optional[Dict[Any, Any]] = None):
        """
        Save the model `state_dict` and its configuration file so that it can be loaded again.

        :param save_dir: The directory in which the model should be saved.
        :param state_dict: A dictionary containing the whole state of the module, including names of layers. By default, the unchanged state dictionary of the module is used.
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        # Save Weights
        save_name = Path(save_dir) / "language_model.bin"
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model  # Only save the model itself

        if not state_dict:
            state_dict = model_to_save.state_dict()  # type: ignore [union-attr]
        torch.save(state_dict, save_name)
        self.save_config(save_dir)

    def formatted_preds(
        self, logits, samples, ignore_first_token: bool = True, padding_mask: Optional[torch.Tensor] = None
    ) -> List[Dict[str, Any]]:
        """
        Extracting vectors from a language model (for example, for extracting sentence embeddings).
        You can use different pooling strategies and layers by specifying them in the object attributes
        `extraction_layer` and `extraction_strategy`. You should set both these attirbutes using the Inferencer:
        Example:  Inferencer(extraction_strategy='cls_token', extraction_layer=-1)

        :param logits: Tuple of (sequence_output, pooled_output) from the language model.
                       Sequence_output: one vector per token, pooled_output: one vector for whole sequence.
        :param samples: For each item in logits, we need additional meta information to format the prediction (for example, input text).
                        This is created by the Processor and passed in here from the Inferencer.
        :param ignore_first_token: When set to `True`, includes the first token for pooling operations (for example, reduce_mean).
                                   Many models use a special token, like [CLS], that you don't want to include in your average of token embeddings.
        :param padding_mask: Mask for the padding tokens. These aren't included in the pooling operations to prevent a bias by the number of padding tokens.
        :param input_ids: IDs of the tokens in the vocabulary.
        :param kwargs: kwargs
        :return: A list of dictionaries containing predictions, for example: [{"context": "some text", "vec": [-0.01, 0.5 ...]}].
        """
        if not hasattr(self, "extraction_layer") or not hasattr(self, "extraction_strategy"):
            raise ModelingError(
                "`extraction_layer` or `extraction_strategy` not specified for LM. "
                "Make sure to set both, e.g. via Inferencer(extraction_strategy='cls_token', extraction_layer=-1)`"
            )

        # unpack the tuple from LM forward pass
        sequence_output = logits[0][0]
        pooled_output = logits[0][1]

        # aggregate vectors
        if self.extraction_strategy == "pooled":
            if self.extraction_layer != -1:
                raise ModelingError(
                    f"Pooled output only works for the last layer, but got extraction_layer={self.extraction_layer}. "
                    "Please set `extraction_layer=-1`"
                )
            vecs = pooled_output.cpu().numpy()

        elif self.extraction_strategy == "per_token":
            vecs = sequence_output.cpu().numpy()

        elif self.extraction_strategy == "reduce_mean":
            vecs = self._pool_tokens(sequence_output, padding_mask, self.extraction_strategy, ignore_first_token=ignore_first_token)
        elif self.extraction_strategy == "reduce_max":
            vecs = self._pool_tokens(sequence_output, padding_mask, self.extraction_strategy, ignore_first_token=ignore_first_token)
        elif self.extraction_strategy == "cls_token":
            vecs = sequence_output[:, 0, :].cpu().numpy()
        else:
            raise NotImplementedError(f"This extraction strategy ({self.extraction_strategy}) is not supported by Haystack.")

        preds = []
        for vec, sample in zip(vecs, samples):
            pred = {}
            pred["context"] = sample.clear_text["text"]
            pred["vec"] = vec
            preds.append(pred)
        return preds

    def _pool_tokens(self, sequence_output: torch.Tensor, padding_mask: torch.Tensor, strategy: str, ignore_first_token: bool):
        token_vecs = sequence_output.cpu().numpy()
        # we only take the aggregated value of non-padding tokens
        padding_mask = padding_mask.cpu().numpy()
        ignore_mask_2d = padding_mask == 0
        # sometimes we want to exclude the CLS token as well from our aggregation operation
        if ignore_first_token:
            ignore_mask_2d[:, 0] = True
        ignore_mask_3d = np.zeros(token_vecs.shape, dtype=bool)
        ignore_mask_3d[:, :, :] = ignore_mask_2d[:, :, np.newaxis]
        if strategy == "reduce_max":
            pooled_vecs = np.ma.array(data=token_vecs, mask=ignore_mask_3d).max(axis=1).data
        if strategy == "reduce_mean":
            pooled_vecs = np.ma.array(data=token_vecs, mask=ignore_mask_3d).mean(axis=1).data

        return pooled_vecs


__all__ = ["ILanguageModel", "IFlow"]
