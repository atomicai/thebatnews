import logging
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Union

import torch
from torch import nn

from thebatnews.modeling.flow import IFlow
from thebatnews.modeling.prime import ILanguageModel
from thebatnews.processing import IProcessor

tracker = None


logger = logging.getLogger(__name__)


class IRunner:
    """
    Base Class for implementing M1/M2/M3 etc... models with frameworks like PyTorch and co.
    """

    subclasses = {}  # type: Dict

    def __init_subclass__(cls, **kwargs):
        """
        This automatically keeps track of all available subclasses.
        Enables generic load() for all specific Mi implementation.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def __init__(self, prediction_heads: Union[List[IFlow], nn.ModuleList]):
        self.prediction_heads = prediction_heads

    @classmethod
    def load(cls, **kwargs):
        """
        Load corresponding M1 or ONNX (not yet) Class(M1/M2/M3l) based on the
        files in the load_dir.

        :param kwargs: Arguments to pass for loading the model.
        :return: Instance of a model.
        """
        if (Path(kwargs["load_dir"]) / "model.onnx").is_file():
            model = cls.subclasses["ONNXM1Runner"].load(**kwargs)
        else:
            model = cls.subclasses["M1Runner"].load(**kwargs)
        return model

    def logits_to_preds(self, logits: torch.Tensor, **kwargs):
        """
        Get predictions from all prediction heads.

        :param logits: Logits that can vary in shape and type, depending on task.
        :return: A list of all predictions from all prediction heads.
        """
        all_preds = []
        # collect preds from all heads
        for head, logits_for_head in zip(self.prediction_heads, logits):
            preds = head.logits_to_preds(logits=logits_for_head, **kwargs)
            all_preds.append(preds)
        return all_preds

    def formatted_preds(self, logits: torch.Tensor, **kwargs):
        """
        Format predictions for inference.

        :param logits: Model logits.
        :return: Predictions in the right format.
        """
        n_heads = len(self.prediction_heads)

        if n_heads == 0:
            # just return LM output (e.g. useful for extracting embeddings at inference time)
            preds_final = self.language_model.formatted_preds(logits=logits, **kwargs)

        elif n_heads == 1:
            preds_final = []
            # This try catch is to deal with the fact that sometimes we collect preds before passing it to
            # formatted_preds (see Inferencer._get_predictions_and_aggregate()) and sometimes we don't
            # (see Inferencer._get_predictions())
            try:
                preds = kwargs["preds"]
                temp = [y[0] for y in preds]
                preds_flat = [item for sublist in temp for item in sublist]
                kwargs["preds"] = preds_flat
            except KeyError:
                kwargs["preds"] = None
            head = self.prediction_heads[0]
            logits_for_head = logits[0]
            preds = head.formatted_preds(logits=logits_for_head, **kwargs)  # type: ignore [operator]
            # TODO This is very messy - we need better definition of what the output should look like
            if type(preds) == list:
                preds_final += preds
            elif type(preds) == dict and "predictions" in preds:
                preds_final.append(preds)

        return preds_final

    def connect_heads_with_processor(self, tasks: Dict, require_labels: bool = True):
        """
        Populates prediction heads (flow) with information coming from tasks.

        :param tasks: A dictionary where the keys are the names of the tasks and
                      the values are the details of the task (e.g. label_list, metric,
                      tensor name).
        :param require_labels: If True, an error will be thrown when a task is
                               not supplied with labels.
        :return: None
        """
        for head in self.prediction_heads:
            head.label_tensor_name = tasks[head.task_name]["label_tensor_name"]
            label_list = tasks[head.task_name]["label_list"]
            if not label_list and require_labels:
                raise Exception(f"The task '{head.task_name}' is missing a valid set of labels")
            label_list = tasks[head.task_name]["label_list"]
            head.label_list = label_list
            head.metric = tasks[head.task_name]["metric"]

    @classmethod
    def _get_prediction_head_files(cls, load_dir: Union[str, Path], strict: bool = True):
        load_dir = Path(load_dir)
        files = os.listdir(load_dir)
        model_files = [load_dir / f for f in files if ".bin" in f and "prediction_head" in f]
        config_files = [load_dir / f for f in files if "config.json" in f and "prediction_head" in f]
        # sort them to get correct order in case of multiple prediction heads
        model_files.sort()
        config_files.sort()

        if strict:
            error_str = (
                f"There is a mismatch in number of model files ({len(model_files)}) and config files"
                f" ({len(config_files)}).This might be because the Language Model Prediction Head does not currently"
                " support saving and loading"
            )
            assert len(model_files) == len(config_files), error_str
        logger.info("Found files for loading %s prediction heads", len(model_files))

        return model_files, config_files


def loss_per_head_sum(loss_per_head: Iterable, global_step: Optional[int] = None, batch: Optional[Dict] = None):
    """
    Sums up the loss of each prediction head.

    :param loss_per_head: List of losses.
    """
    return sum(loss_per_head)


class M1Runner(nn.Module, IRunner):
    """
    PyTorch implementation containing all the modelling needed for your NLP task. Combines a language
    model and a prediction head. Allows for gradient flow back to the language model component.
    """

    def __init__(
        self,
        language_model: ILanguageModel,
        prediction_heads: List[IFlow],
        embeds_dropout_prob: float,
        lm_output_types: Union[str, List[str]],
        device: torch.device,
        loss_aggregation_fn: Optional[Callable] = None,
    ):
        """
        :param language_model: Any model that turns token ids into vector representations.
        :param prediction_heads: A list of models that take embeddings and return logits for a given task.
        :param embeds_dropout_prob: The probability that a value in the embeddings returned by the
                                    language model will be zeroed.
        :param lm_output_types: How to extract the embeddings from the final layer of the language model. When set
                                to "per_token", one embedding will be extracted per input token. If set to
                                "per_sequence", a single embedding will be extracted to represent the full
                                input sequence. Can either be a single string, or a list of strings,
                                one for each prediction head.
        :param device: The device on which this model will operate. Either torch.device("cpu") or torch.device("cuda").
        :param loss_aggregation_fn: Function to aggregate the loss of multiple prediction heads.
                                    Input: loss_per_head (list of tensors), global_step (int), batch (dict)
                                    Output: aggregated loss (tensor)
                                    Default is a simple sum:
                                    `lambda loss_per_head, global_step=None, batch=None: sum(tensors)`
                                    However, you can pass more complex functions that depend on the
                                    current step (e.g. for round-robin style multitask learning) or the actual
                                    content of the batch (e.g. certain labels)
                                    Note: The loss at this stage is per sample, i.e one tensor of
                                    shape (batchsize) per prediction head.
        """
        super(M1Runner, self).__init__()  # type: ignore
        self.device = device
        self.language_model = language_model.to(device)
        self.lm_output_dims = language_model.output_dims
        self.prediction_heads = nn.ModuleList([ph.to(device) for ph in prediction_heads])
        self.fit_heads_to_lm()
        self.dropout = nn.Dropout(embeds_dropout_prob)
        self.lm_output_types = [lm_output_types] if isinstance(lm_output_types, str) else lm_output_types
        self.log_params()
        # default loss aggregation function is a simple sum (without using any of the optional params)
        if not loss_aggregation_fn:
            loss_aggregation_fn = loss_per_head_sum
        self.loss_aggregation_fn = loss_aggregation_fn

    def fit_heads_to_lm(self):
        """
        This iterates over each prediction head and ensures that its input
        dimensionality matches the output dimensionality of the language model.
        If it doesn't, it is resized so it does fit.
        """
        for ph in self.prediction_heads:
            ph.resize_input(self.lm_output_dims)
            ph.to(self.device)

    def bypass_ph(self):
        """
        Replaces methods in the prediction heads with dummy functions.
        Used for benchmarking where we want to isolate the LanguageModel run time
        from the PredictionHead run time.
        """
        # TODO convert inner functions into lambdas

        def fake_forward(x):
            """
            Slices lm vector outputs of shape (batch_size, max_seq_len, dims) --> (batch_size, max_seq_len, 2)
            """
            return x.narrow(2, 0, 2)

        def fake_logits_to_preds(logits, **kwargs):
            batch_size = logits.shape[0]
            return [None, None] * batch_size

        def fake_formatted_preds(**kwargs):
            return None

        for ph in self.prediction_heads:
            ph.forward = fake_forward
            ph.logits_to_preds = fake_logits_to_preds
            ph.formatted_preds = fake_formatted_preds

    def save(self, save_dir: Union[str, Path]):
        """
        Saves the language model and prediction heads. This will generate a config file
        and model weights for each.

        :param save_dir: Path to save the M1 model to
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.language_model.save(save_dir)
        for i, ph in enumerate(self.prediction_heads):
            ph.save(str(save_dir), i)
            # Need to save config and pipeline

    @classmethod
    def load(  # type: ignore
        cls,
        load_dir: Union[str, Path],
        device: Union[str, torch.device],
        strict: bool = True,
        processor: Optional[IProcessor] = None,
    ):
        """
        Loads M1 from a directory. The directory must contain:

        * language_model.bin
        * language_model_config.json
        * prediction_head_X.bin  multiple PH possible
        * prediction_head_X_config.json
        * processor_config.json config for transforming input
        * vocab.txt vocab file for language model, turning text to Wordpiece Tokens

        :param load_dir: Location where the AdaptiveModel is stored.
        :param device: Specifies the device to which you want to send the model, either torch.device("cpu") or torch.device("cuda").
        :param strict: Whether to strictly enforce that the keys loaded from saved model match the ones in
                       the PredictionHead (see torch.nn.module.load_state_dict()).
        :param processor: Processor to populate prediction head with information coming from tasks.
        """
        device = torch.device(device)
        config_file = Path(load_dir) / "language_model_config.json"
        assert config_file.exists(), "The config is not found, couldn't load the model"
        language_model = ILanguageModel.load(load_dir)
        # Prediction heads
        _, ph_config_files = cls._get_prediction_head_files(load_dir)
        prediction_heads = []
        ph_output_type = []
        for config_file in ph_config_files:
            head = IFlow.load(config_file, strict=strict)
            prediction_heads.append(head)
            ph_output_type.append(head.ph_output_type)

        model = cls(language_model, prediction_heads, 0.1, ph_output_type, device)
        if processor:
            model.connect_heads_with_processor(processor.tasks)

        return model

    def logits_to_loss_per_head(self, logits: torch.Tensor, **kwargs):
        """
        Collect losses from each prediction head.

        :param logits: Logits, can vary in shape and type, depending on task.
        :return: The per sample per prediciton head loss whose first two dimensions
                 have length n_pred_heads, batch_size.
        """
        all_losses = []
        for head, logits_for_one_head in zip(self.prediction_heads, logits):
            # check if PredictionHead connected to Processor
            assert hasattr(head, "label_tensor_name"), (
                f"Label_tensor_names are missing inside the {head.task_name} Prediction Head. Did you connect the model"
                " with the processor through either 'model.connect_heads_with_processor(processor.tasks)'"
                " or by passing the processor to the Adaptive Model?"
            )
            all_losses.append(head.logits_to_loss(logits=logits_for_one_head, **kwargs))
        return all_losses

    def logits_to_loss(self, logits: torch.Tensor, global_step: Optional[int] = None, **kwargs):
        """
        Get losses from all prediction heads & reduce to single loss *per sample*.

        :param logits: Logits, can vary in shape and type, depending on task.
        :param global_step: Number of current training step.
        :param kwargs: Placeholder for passing generic parameters.
                       Note: Contains the batch (as dict of tensors), when called from Trainer.train().
        :return: torch.tensor that is the per sample loss (len: batch_size)
        """
        all_losses = self.logits_to_loss_per_head(logits, **kwargs)
        # This aggregates the loss per sample across multiple prediction heads
        # Default is sum(), but you can configure any fn that takes [Tensor, Tensor ...] and returns [Tensor]
        loss = self.loss_aggregation_fn(all_losses, global_step=global_step, batch=kwargs)
        return loss

    def prepare_labels(self, **kwargs):
        """
        Label conversion to original label space, per prediction head.

        :param label_maps: dictionary for mapping ids to label strings
        :type label_maps: dict[int:str]
        :return: labels in the right format
        """
        all_labels = []
        # for head, label_map_one_head in zip(self.prediction_heads):
        #     labels = head.prepare_labels(label_map=label_map_one_head, **kwargs)
        #     all_labels.append(labels)
        for head in self.prediction_heads:
            labels = head.prepare_labels(**kwargs)
            all_labels.append(labels)
        return all_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs
    ):
        """
        Push data through the whole model and returns logits. The data will
        propagate through the language model and each of the attached prediction heads.

        :param input_ids: The IDs of each token in the input sequence. It's a tensor of shape [batch_size, max_seq_len].
        :param segment_ids: The ID of the segment. For example, in next sentence prediction, the tokens in the
           first sentence are marked with 0 and the tokens in the second sentence are marked with 1.
           It is a tensor of shape [batch_size, max_seq_len].
        :param padding_mask: A mask that assigns 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len].
        :param output_hidden_states: Whether to output hidden states
        :param output_attentions: Whether to output attentions
        :return: All logits as torch.tensor or multiple tensors.
        """
        # Run forward pass of language model
        output_tuple = self.language_model.forward(
            input_ids=input_ids,
            segment_ids=segment_ids,
            attention_mask=padding_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        if output_hidden_states and output_attentions:
            sequence_output, pooled_output, hidden_states, attentions = output_tuple
        elif output_hidden_states:
            sequence_output, pooled_output, hidden_states = output_tuple
        elif output_attentions:
            sequence_output, pooled_output, attentions = output_tuple
        else:
            sequence_output, pooled_output = output_tuple
        # Run forward pass of (multiple) prediction heads using the output from above
        all_logits = []
        if len(self.prediction_heads) > 0:
            for head, lm_out in zip(self.prediction_heads, self.lm_output_types):
                # Choose relevant vectors from LM as output and perform dropout
                if lm_out == "per_token":
                    output = self.dropout(sequence_output)
                elif lm_out == "per_sequence" or lm_out == "per_sequence_continuous":
                    output = self.dropout(pooled_output)
                else:
                    raise ValueError("Unknown extraction strategy from language model: {}".format(lm_out))

                # Do the actual forward pass of a single head
                all_logits.append(head(output))
        else:
            # just return LM output (e.g. useful for extracting embeddings at inference time)
            all_logits.append((sequence_output, pooled_output))

        if output_hidden_states and output_attentions:
            return all_logits, hidden_states, attentions
        if output_hidden_states:
            return all_logits, hidden_states
        if output_attentions:
            return all_logits, attentions
        return all_logits

    def forward_lm(self, **kwargs):
        """
        Forward pass for the language model.

        :return: Tuple containing list of embeddings for each token and
                 embedding for whole sequence.
        """
        # Check if we have to extract from a special layer of the LM (default = last layer)
        try:
            extraction_layer = self.language_model.extraction_layer
        except:
            extraction_layer = -1

        # Run forward pass of language model
        if extraction_layer == -1:
            sequence_output, pooled_output = self.language_model(
                **kwargs, return_dict=False, output_all_encoded_layers=False
            )
        else:
            # get output from an earlier layer
            self.language_model.enable_hidden_states_output()
            sequence_output, pooled_output, all_hidden_states = self.language_model(**kwargs, return_dict=False)
            sequence_output = all_hidden_states[extraction_layer]
            pooled_output = None  # not available in earlier layers
            self.language_model.disable_hidden_states_output()
        return sequence_output, pooled_output

    def log_params(self):
        """
        Logs parameters to generic logger MlLogger
        """
        params = {
            "lm_type": self.language_model.__class__.__name__,
            "lm_name": self.language_model.name,
            "prediction_heads": ",".join([head.__class__.__name__ for head in self.prediction_heads]),
            "lm_output_types": ",".join(self.lm_output_types),
        }
        try:
            if tracker is not None:
                tracker.track_params(params)
        except Exception as e:
            logger.warning("ML logging didn't work: %s", e)

    def verify_vocab_size(self, vocab_size: int):
        """
        Verifies that the model fits to the tokenizer vocabulary.
        They could diverge in case of custom vocabulary added via tokenizer.add_tokens()
        """
        model_vocab_len = self.language_model.model.resize_token_embeddings(new_num_tokens=None).num_embeddings  # type: ignore [union-attr,operator]

        msg = (
            f"Vocab size of tokenizer {vocab_size} doesn't match with model {model_vocab_len}. "
            "If you added a custom vocabulary to the tokenizer, "
            "make sure to supply 'n_added_tokens' to get_language_model() and BertStyleLM.load()"
        )
        assert vocab_size == model_vocab_len, msg

        for head in self.prediction_heads:
            if head.model_type == "language_modelling":
                ph_decoder_len = head.decoder.weight.shape[0]  # type: ignore [union-attr,index]
                assert vocab_size == ph_decoder_len, msg

    def get_language(self):
        return self.language_model.language


__all__ = ["M1Runner"]
