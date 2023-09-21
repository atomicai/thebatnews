import logging
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification
from transformers.models.bert.modeling_bert import ACT2FN, BertForPreTraining

from thebatnews.etc.format import convert_iob_to_simple_tags
from thebatnews.modeling import div
from thebatnews.modeling.loss import Losses
from thebatnews.modeling.mask import IFlow

logger = logging.getLogger(__name__)

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except (ImportError, AttributeError) as e:
    logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    BertLayerNorm = torch.nn.LayerNorm


class IRegressionHead(IFlow):
    def __init__(
        self,
        layer_dims=[768, 1],
        task_name="regression",
        **kwargs,
    ):
        super(IRegressionHead, self).__init__()
        # num_labels could in most cases also be automatically retrieved from the data processor
        self.layer_dims = layer_dims
        self.feed_forward = div.FeedForwardBlock(self.layer_dims)
        # num_labels is being set to 2 since it is being hijacked to store the scaling factor and the mean
        self.num_labels = 2
        self.ph_output_type = "per_sequence_continuous"
        self.model_type = "regression"
        self.loss_fct = Losses.CrossEntropyLoss(reduction="none")
        self.task_name = task_name
        self.generate_config()

    def forward(self, x):
        logits = self.feed_forward(x)
        return logits

    def logits_to_loss(self, logits, **kwargs):
        label_ids = kwargs.get(self.label_tensor_name)
        return self.loss_fct(logits, label_ids.float())

    def logits_to_preds(self, logits, **kwargs):
        preds = logits.cpu().numpy()
        # rescale predictions to actual label distribution
        preds = [x * self.label_list[1] + self.label_list[0] for x in preds]
        return preds

    def prepare_labels(self, **kwargs):
        label_ids = kwargs.get(self.label_tensor_name)
        label_ids = label_ids.cpu().numpy()
        label_ids = [x * self.label_list[1] + self.label_list[0] for x in label_ids]
        return label_ids

    def formatted_preds(self, logits, samples, **kwargs):
        preds = self.logits_to_preds(logits)
        contexts = [sample.clear_text["text"] for sample in samples]

        res = {"task": "regression", "predictions": []}
        for pred, context in zip(preds, contexts):
            res["predictions"].append({"context": f"{context}", "pred": pred[0]})
        return res


class ICLSHead(IFlow):
    def __init__(
        self,
        layer_dims=None,
        num_labels=None,
        class_weights=None,
        loss_ignore_index=-100,
        loss_reduction="none",
        task_name="text_classification",
        **kwargs,
    ):
        """
        :param layer_dims: The size of the layers in the feed forward component. The feed forward will have as many layers as there are ints in this list. This param will be deprecated in future
        :type layer_dims: list
        :param num_labels: The numbers of labels. Use to set the size of the final layer in the feed forward component. It is recommended to only set num_labels or layer_dims, not both.
        :type num_labels: int
        :param class_weights:
        :param loss_ignore_index:
        :param loss_reduction:
        :param task_name:
        :param kwargs:
        """
        super(ICLSHead, self).__init__()
        # num_labels could in most cases also be automatically retrieved from the data processor
        if layer_dims:
            self.layer_dims = layer_dims
            logger.warning("`layer_dims` will be deprecated in future releases")
        elif num_labels:
            self.layer_dims = [768, num_labels]
        else:
            raise ValueError("Please supply `num_labels` to define output dim of prediction head")
        self.num_labels = self.layer_dims[-1]
        self.feed_forward = div.FeedForwardBlock(self.layer_dims)
        logger.info(f"Prediction head initialized with size {self.layer_dims}")
        self.num_labels = self.layer_dims[-1]
        self.ph_output_type = "per_sequence"
        self.model_type = "text_classification"
        self.task_name = task_name  # used for connecting with the right output of the processor

        if type(class_weights) is np.ndarray and class_weights.ndim != 1:
            raise ValueError(
                "When you pass `class_weights` as `np.ndarray` it must have 1 dimension! You provided {} dimensions.".format(
                    class_weights.ndim
                )
            )

        self.class_weights = class_weights

        if class_weights is not None:
            logger.info(f"Using class weights for task '{self.task_name}': {self.class_weights}")
            balanced_weights = nn.Parameter(torch.tensor(class_weights), requires_grad=False)
        else:
            balanced_weights = None

        self.loss_fct = Losses.FocalLoss(
            weight=balanced_weights,
            # reduction=loss_reduction,
            # ignore_index=loss_ignore_index,
        )

        # add label list
        if "label_list" in kwargs:
            self.label_list = kwargs["label_list"]

        self.generate_config()

    @classmethod
    def load(cls, pretrained_model_name_or_path, revision=None):
        """
        Load a prediction head from a saved FARM or transformers model. `pretrained_model_name_or_path`
        can be one of the following:
        a) Local path to a FARM prediction head config (e.g. my-bert/prediction_head_0_config.json)
        b) Local path to a Transformers model (e.g. my-bert)
        c) Name of a public model from https://huggingface.co/models (e.g. distilbert-base-uncased-distilled-squad)


        :param pretrained_model_name_or_path: local path of a saved model or name of a publicly available model.
                                              Exemplary public name:
                                              - deepset/bert-base-german-cased-hatespeech-GermEval18Coarse

                                              See https://huggingface.co/models for full list
        :param revision: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :type revision: str

        """

        if (
            os.path.exists(pretrained_model_name_or_path)
            and "config.json" in pretrained_model_name_or_path
            and "prediction_head" in pretrained_model_name_or_path
        ):
            # a) FARM style
            head = super(ICLSHead, cls).load(pretrained_model_name_or_path)
        else:
            # b) transformers style
            # load all weights from model
            full_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path, revision=revision)
            # init empty head
            head = cls(layer_dims=[full_model.config.hidden_size, len(full_model.config.id2label)])
            # transfer weights for head from full model
            head.feed_forward.feed_forward[0].load_state_dict(full_model.classifier.state_dict())
            # add label list
            head.label_list = list(full_model.config.id2label.values())
            del full_model

        return head

    def forward(self, X):
        logits = self.feed_forward(X)
        return logits

    def logits_to_loss(self, logits, **kwargs):
        label_ids = kwargs.get(self.label_tensor_name)
        label_ids = label_ids
        return self.loss_fct(logits, label_ids.view(-1))

    def logits_to_probs(self, logits, return_class_probs: bool = True, **kwargs):
        softmax = torch.nn.Softmax(dim=1)
        probs = softmax(logits)
        if return_class_probs:
            probs = probs
        else:
            probs = torch.max(probs, dim=1)[0]
        probs = probs.cpu().numpy()
        return probs

    def logits_to_preds(self, logits, return_pred_ids: bool = False, **kwargs):
        logits = logits.cpu().numpy()
        pred_ids = logits.argmax(1)
        preds = [self.label_list[int(x)] for x in pred_ids]
        if return_pred_ids:
            return preds, pred_ids
        return preds

    def prepare_labels(self, **kwargs):
        label_ids = kwargs.get(self.label_tensor_name)
        label_ids = label_ids.cpu().numpy()
        # This is the standard doc classification case
        try:
            labels = [self.label_list[int(x)] for x in label_ids]
        # This case is triggered in Natural Questions where each example can have multiple labels
        except TypeError:
            labels = [self.label_list[int(x[0])] for x in label_ids]
        return labels

    def formatted_preds(self, logits=None, preds=None, samples=None, return_class_probs=False, **kwargs):
        """Like QuestionAnsweringHead.formatted_preds(), this fn can operate on either logits or preds. This
        is needed since at inference, the order of operations is very different depending on whether we are performing
        aggregation or not (compare Inferencer._get_predictions() vs Inferencer._get_predictions_and_aggregate())"""

        assert (logits is not None) or (preds is not None)

        # When this method is used along side a QAHead at inference (e.g. Natural Questions), preds is the input and
        # there is currently no good way of generating probs
        if logits is not None:
            preds = self.logits_to_preds(logits)
            probs = self.logits_to_probs(logits, return_class_probs)
        else:
            probs = [None] * len(preds)

        # TODO this block has to do with the difference in Basket and Sample structure between SQuAD and NQ
        try:
            contexts = [sample.clear_text["text"] for sample in samples]
        # This case covers Natural Questions where the sample is in a QA style
        except KeyError:
            contexts = [sample.clear_text["question_text"] + " | " + sample.clear_text["passage_text"] for sample in samples]

        contexts_b = [sample.clear_text["text_b"] for sample in samples if "text_b" in sample.clear_text]
        if len(contexts_b) != 0:
            contexts = ["|".join([a, b]) for a, b in zip(contexts, contexts_b)]

        res = {"task": "text_classification", "predictions": []}
        for pred, prob, context in zip(preds, probs, contexts):
            if not return_class_probs:
                pred_dict = {
                    "start": None,
                    "end": None,
                    "context": f"{context}",
                    "label": f"{pred}",
                    "probability": prob,
                }
            else:
                pred_dict = {
                    "start": None,
                    "end": None,
                    "context": f"{context}",
                    "label": "class_probabilities",
                    "probability": prob,
                }

            res["predictions"].append(pred_dict)
        return res


class IMLCLSHead(IFlow):
    def __init__(
        self,
        layer_dims=None,
        num_labels=None,
        class_weights=None,
        loss_reduction="none",
        task_name="text_classification",
        pred_threshold=0.5,
        **kwargs,
    ):
        """
        :param layer_dims: The size of the layers in the feed forward component. The feed forward will have as many layers as there are ints in this list. This param will be deprecated in future
        :type layer_dims: list
        :param num_labels: The numbers of labels. Use to set the size of the final layer in the feed forward component. It is recommended to only set num_labels or layer_dims, not both.
        :type num_labels: int
        :param class_weights:
        :param loss_reduction:
        :param task_name:
        :param pred_threshold:
        :param kwargs:
        """
        super(IMLCLSHead, self).__init__()
        # num_labels could in most cases also be automatically retrieved from the data processor
        if layer_dims:
            self.layer_dims = layer_dims
            logger.warning("`layer_dims` will be deprecated in future releases")
        elif num_labels:
            self.layer_dims = [768, num_labels]
        else:
            raise ValueError("Please supply `num_labels` to define output dim of prediction head")
        self.num_labels = self.layer_dims[-1]
        logger.info(f"Prediction head initialized with size {self.layer_dims}")
        self.feed_forward = div.FeedForwardBlock(self.layer_dims)
        self.ph_output_type = "per_sequence"
        self.model_type = "multilabel_text_classification"
        self.task_name = task_name  # used for connecting with the right output of the processor
        self.class_weights = class_weights
        self.pred_threshold = pred_threshold

        if class_weights is not None:
            logger.info(f"Using class weights for task '{self.task_name}': {self.class_weights}")
            # TODO must balanced weight really be a instance attribute?
            self.balanced_weights = nn.Parameter(torch.tensor(class_weights), requires_grad=False)
        else:
            self.balanced_weights = None

        self.loss_fct = BCEWithLogitsLoss(pos_weight=self.balanced_weights, reduction=loss_reduction)

        self.generate_config()

    def forward(self, X):
        logits = self.feed_forward(X)
        return logits

    def logits_to_loss(self, logits, **kwargs):
        label_ids = kwargs.get(self.label_tensor_name).to(dtype=torch.float)
        loss = self.loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1, self.num_labels))
        per_sample_loss = loss.mean(1)
        return per_sample_loss

    def logits_to_probs(self, logits, **kwargs):
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits)
        probs = probs.cpu().numpy()
        return probs

    def logits_to_preds(self, logits, **kwargs):
        probs = self.logits_to_probs(logits)
        # TODO we could potentially move this to GPU to speed it up
        pred_ids = [np.where(row > self.pred_threshold)[0] for row in probs]
        preds = []
        for row in pred_ids:
            preds.append([self.label_list[int(x)] for x in row])
        return preds

    def prepare_labels(self, **kwargs):
        label_ids = kwargs.get(self.label_tensor_name)
        label_ids = label_ids.cpu().numpy()
        label_ids = [np.where(row == 1)[0] for row in label_ids]
        labels = []
        for row in label_ids:
            labels.append([self.label_list[int(x)] for x in row])
        return labels

    def formatted_preds(self, logits, samples, **kwargs):
        preds = self.logits_to_preds(logits)
        probs = self.logits_to_probs(logits)
        contexts = [sample.clear_text["text"] for sample in samples]

        res = {"task": "text_classification", "predictions": []}
        for pred, prob, context in zip(preds, probs, contexts):
            res["predictions"].append(
                {
                    "start": None,
                    "end": None,
                    "context": f"{context}",
                    "label": f"{pred}",
                    "probability": prob,
                }
            )
        return res


class ITOKENCLSHead(IFlow):
    def __init__(self, layer_dims=None, num_labels=None, task_name="ner", **kwargs):
        """
        :param layer_dims: The size of the layers in the feed forward component. The feed forward will have as many layers as there are ints in this list. This param will be deprecated in future
        :type layer_dims: list
        :param num_labels: The numbers of labels. Use to set the size of the final layer in the feed forward component. It is recommended to only set num_labels or layer_dims, not both.
        :type num_labels: int
        :param task_name:
        :param kwargs:
        """
        super(ITOKENCLSHead, self).__init__()
        if layer_dims:
            self.layer_dims = layer_dims
            logger.warning("`layer_dims` will be deprecated in future releases")
        elif num_labels:
            self.layer_dims = [768, num_labels]
        else:
            raise ValueError("Please supply `num_labels` to define output dim of prediction head")
        self.num_labels = self.layer_dims[-1]
        logger.info(f"Prediction head initialized with size {self.layer_dims}")
        self.feed_forward = div.FeedForwardBlock(self.layer_dims)
        self.num_labels = self.layer_dims[-1]
        self.loss_fct = Losses.CrossEntropyLoss(reduction="none")
        self.ph_output_type = "per_token"
        self.model_type = "token_classification"
        self.task_name = task_name
        if "label_list" in kwargs:
            self.label_list = kwargs["label_list"]
        self.generate_config()

    @classmethod
    def load(cls, pretrained_model_name_or_path, revision=None):
        """
        Load a prediction head from a saved FARM or transformers model. `pretrained_model_name_or_path`
        can be one of the following:
        a) Local path to a FARM prediction head config (e.g. my-bert/prediction_head_0_config.json)
        b) Local path to a Transformers model (e.g. my-bert)
        c) Name of a public model from https://huggingface.co/models (e.g.bert-base-cased-finetuned-conll03-english)


        :param pretrained_model_name_or_path: local path of a saved model or name of a publicly available model.
                                              Exemplary public names:
                                              - bert-base-cased-finetuned-conll03-english

                                              See https://huggingface.co/models for full list

        """

        if (
            os.path.exists(pretrained_model_name_or_path)
            and "config.json" in pretrained_model_name_or_path
            and "prediction_head" in pretrained_model_name_or_path
        ):
            # a) FARM style
            head = super(ITOKENCLSHead, cls).load(pretrained_model_name_or_path)
        else:
            # b) transformers style
            # load all weights from model
            full_model = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path, revision=revision)
            # init empty head
            head = cls(layer_dims=[full_model.config.hidden_size, len(full_model.config.label2id)])
            # transfer weights for head from full model
            head.feed_forward.feed_forward[0].load_state_dict(full_model.classifier.state_dict())
            # add label list
            head.label_list = list(full_model.config.id2label.values())
            head.generate_config()
            del full_model
        return head

    def forward(self, X):
        logits = self.feed_forward(X)
        return logits

    def logits_to_loss(self, logits, initial_mask, padding_mask=None, **kwargs):
        label_ids = kwargs.get(self.label_tensor_name)

        # Todo: should we be applying initial mask here? Loss is currently calculated even on non initial tokens
        active_loss = padding_mask.view(-1) == 1
        active_logits = logits.view(-1, self.num_labels)[active_loss]
        active_labels = label_ids.view(-1)[active_loss]

        loss = self.loss_fct(active_logits, active_labels)  # loss is a 1 dimemnsional (active) token loss
        return loss

    def logits_to_preds(self, logits, initial_mask, **kwargs):
        preds_word_all = []
        preds_tokens = torch.argmax(logits, dim=2)
        preds_token = preds_tokens.detach().cpu().numpy()
        # used to be: padding_mask = padding_mask.detach().cpu().numpy()
        initial_mask = initial_mask.detach().cpu().numpy()

        for idx, im in enumerate(initial_mask):
            preds_t = preds_token[idx]
            # Get labels and predictions for just the word initial tokens
            preds_word_id = self.initial_token_only(preds_t, initial_mask=im)
            preds_word = [self.label_list[pwi] for pwi in preds_word_id]
            preds_word_all.append(preds_word)
        return preds_word_all

    def logits_to_probs(self, logits, initial_mask, return_class_probs, **kwargs):
        # get per token probs
        softmax = torch.nn.Softmax(dim=2)
        token_probs = softmax(logits)
        if return_class_probs:
            token_probs = token_probs
        else:
            token_probs = torch.max(token_probs, dim=2)[0]
        token_probs = token_probs.cpu().numpy()

        # convert to per word probs
        all_probs = []
        initial_mask = initial_mask.detach().cpu().numpy()
        for idx, im in enumerate(initial_mask):
            probs_t = token_probs[idx]
            probs_words = self.initial_token_only(probs_t, initial_mask=im)
            all_probs.append(probs_words)
        return all_probs

    def prepare_labels(self, initial_mask, **kwargs):
        label_ids = kwargs.get(self.label_tensor_name)
        labels_all = []
        label_ids = label_ids.cpu().numpy()
        for label_ids_one_sample, initial_mask_one_sample in zip(label_ids, initial_mask):
            label_ids = self.initial_token_only(label_ids_one_sample, initial_mask_one_sample)
            labels = [self.label_list[l] for l in label_ids]
            labels_all.append(labels)
        return labels_all

    @staticmethod
    def initial_token_only(seq, initial_mask):
        ret = []
        for init, s in zip(initial_mask, seq):
            if init:
                ret.append(s)
        return ret

    def formatted_preds(self, logits, initial_mask, samples, return_class_probs=False, **kwargs):
        preds = self.logits_to_preds(logits, initial_mask)
        probs = self.logits_to_probs(logits, initial_mask, return_class_probs)

        # align back with original input by getting the original word spans
        spans = [s.tokenized["word_spans"] for s in samples]
        res = {"task": "ner", "predictions": []}
        for preds_seq, probs_seq, sample, spans_seq in zip(preds, probs, samples, spans):
            tags, spans_seq, tag_probs = convert_iob_to_simple_tags(preds_seq, spans_seq, probs_seq)
            seq_res = []
            # TODO: Though we filter out tags and spans for non-entity words,
            # TODO: we do not yet filter out probs of non-entity words. This needs to be implemented still
            for tag, tag_prob, span in zip(tags, tag_probs, spans_seq):
                context = sample.clear_text["text"][span[0] : span[1]]
                seq_res.append(
                    {
                        "start": span[0],
                        "end": span[1],
                        "context": f"{context}",
                        "label": f"{tag}",
                        "probability": tag_prob,
                    }
                )
            res["predictions"].append(seq_res)
        return res


class LMHead(IFlow):
    def __init__(self, hidden_size, vocab_size, hidden_act="gelu", task_name="lm", **kwargs):
        super(LMHead, self).__init__()

        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.vocab_size = vocab_size
        self.loss_fct = Losses.CrossEntropyLoss(reduction="none", ignore_index=-1)
        self.num_labels = vocab_size  # vocab size
        # Adding layer_dims (required for conversion to transformers)
        self.layer_dims = [hidden_size, vocab_size]
        # TODO Check if weight init needed!
        # self.apply(self.init_bert_weights)
        self.ph_output_type = "per_token"

        self.model_type = "language_modelling"
        self.task_name = task_name
        self.generate_config()

        # NN Layers
        # this is the "transform" module in the pytorch-transformers repo
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.transform_act_fn = ACT2FN[self.hidden_act]
        self.LayerNorm = BertLayerNorm(self.hidden_size, eps=1e-12)

        # this is the "decoder" in the pytorch-transformers repo
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    @classmethod
    def load(cls, pretrained_model_name_or_path, revision=None, n_added_tokens=0):
        """
        Load a prediction head from a saved FARM or transformers model. `pretrained_model_name_or_path`
        can be one of the following:
        a) Local path to a FARM prediction head config (e.g. my-bert/prediction_head_0_config.json)
        b) Local path to a Transformers model (e.g. my-bert)
        c) Name of a public model from https://huggingface.co/models (e.g.bert-base-cased)


        :param pretrained_model_name_or_path: local path of a saved model or name of a publicly available model.
                                              Exemplary public names:
                                              - bert-base-cased

                                              See https://huggingface.co/models for full list
        :param revision: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :type revision: str

        """

        if (
            os.path.exists(pretrained_model_name_or_path)
            and "config.json" in pretrained_model_name_or_path
            and "prediction_head" in pretrained_model_name_or_path
        ):
            # a) FARM style
            if n_added_tokens != 0:
                # TODO resize prediction head decoder for custom vocab
                raise NotImplementedError("Custom vocab not yet supported for model loading from FARM files")

            head = super(LMHead, cls).load(pretrained_model_name_or_path)
        else:
            # b) pytorch-transformers style
            # load weights from bert model
            # (we might change this later to load directly from a state_dict to generalize for other language models)
            bert_with_lm = BertForPreTraining.from_pretrained(pretrained_model_name_or_path, revision=revision)

            # init empty head
            vocab_size = bert_with_lm.config.vocab_size + n_added_tokens

            head = cls(
                hidden_size=bert_with_lm.config.hidden_size,
                vocab_size=vocab_size,
                hidden_act=bert_with_lm.config.hidden_act,
            )

            # load weights
            head.dense.load_state_dict(bert_with_lm.cls.predictions.transform.dense.state_dict())
            head.LayerNorm.load_state_dict(bert_with_lm.cls.predictions.transform.LayerNorm.state_dict())

            # Not loading weights of decoder here, since we later share weights with the embedding layer of LM
            # head.decoder.load_state_dict(bert_with_lm.cls.predictions.decoder.state_dict())

            if n_added_tokens == 0:
                bias_params = bert_with_lm.cls.predictions.bias
            else:
                # Custom vocab => larger vocab => larger dims of output layer in the LM head
                bias_params = torch.nn.Parameter(torch.cat([bert_with_lm.cls.predictions.bias, torch.zeros(n_added_tokens)]))
            head.bias.data.copy_(bias_params)
            del bert_with_lm
            del bias_params

        return head

    def set_shared_weights(self, shared_embedding_weights):
        self.decoder.weight = shared_embedding_weights

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        lm_logits = self.decoder(hidden_states) + self.bias
        return lm_logits

    def logits_to_loss(self, logits, **kwargs):
        lm_label_ids = kwargs.get(self.label_tensor_name)
        batch_size = lm_label_ids.shape[0]
        masked_lm_loss = self.loss_fct(logits.view(-1, self.num_labels), lm_label_ids.view(-1))
        per_sample_loss = masked_lm_loss.view(-1, batch_size).mean(dim=0)
        return per_sample_loss

    def logits_to_preds(self, logits, **kwargs):
        lm_label_ids = kwargs.get(self.label_tensor_name).cpu().numpy()
        lm_preds_ids = logits.argmax(2).cpu().numpy()
        # apply mask to get rid of predictions for non-masked tokens
        lm_preds_ids[lm_label_ids == -1] = -1
        lm_preds_ids = lm_preds_ids.tolist()
        preds = []
        # we have a batch of sequences here. we need to convert for each token in each sequence.
        for pred_ids_for_sequence in lm_preds_ids:
            preds.append([self.label_list[int(x)] for x in pred_ids_for_sequence if int(x) != -1])
        return preds

    def prepare_labels(self, **kwargs):
        label_ids = kwargs.get(self.label_tensor_name)
        label_ids = label_ids.cpu().numpy().tolist()
        labels = []
        # we have a batch of sequences here. we need to convert for each token in each sequence.
        for ids_for_sequence in label_ids:
            labels.append([self.label_list[int(x)] for x in ids_for_sequence if int(x) != -1])
        return labels


class NSPHead(ICLSHead):
    """
    Almost identical to a TextClassificationHead. Only difference: we can load the weights from
     a pretrained language model that was saved in the Transformers style (all in one model).
    """

    @classmethod
    def load(cls, pretrained_model_name_or_path):
        """
        Load a prediction head from a saved FARM or transformers model. `pretrained_model_name_or_path`
        can be one of the following:
        a) Local path to a FARM prediction head config (e.g. my-bert/prediction_head_0_config.json)
        b) Local path to a Transformers model (e.g. my-bert)
        c) Name of a public model from https://huggingface.co/models (e.g.bert-base-cased)


        :param pretrained_model_name_or_path: local path of a saved model or name of a publicly available model.
                                              Exemplary public names:
                                              - bert-base-cased

                                              See https://huggingface.co/models for full list

        """
        if (
            os.path.exists(pretrained_model_name_or_path)
            and "config.json" in pretrained_model_name_or_path
            and "prediction_head" in pretrained_model_name_or_path
        ):
            # a) FARM style
            head = super(NSPHead, cls).load(pretrained_model_name_or_path)
        else:
            # b) pytorch-transformers style
            # load weights from bert model
            # (we might change this later to load directly from a state_dict to generalize for other language models)
            bert_with_lm = BertForPreTraining.from_pretrained(pretrained_model_name_or_path)

            # init empty head
            head = cls(layer_dims=[bert_with_lm.config.hidden_size, 2], loss_ignore_index=-1, task_name="nextsentence")

            # load weights
            head.feed_forward.feed_forward[0].load_state_dict(bert_with_lm.cls.seq_relationship.state_dict())
            del bert_with_lm

        return head


class IANNHead(IFlow):
    def __init__(self, similarity_function: str = "dot_product", global_loss_buffer_size: int = 150000, **kwargs):
        """
        Init the TextSimilarityHead.

        :param similarity_function: Function to calculate similarity between queries and passage embeddings.
                                    Choose either "dot_product" (Default) or "cosine".
        :param global_loss_buffer_size: Buffer size for all_gather() in DDP.
                                        Increase if errors like "encoded data exceeds max_size ..." come up

        :param kwargs:
        """

        super(IANNHead, self).__init__()

        self.similarity_function = similarity_function
        self.loss_fct = Losses.FocalLoss(reduction="mean")
        self.task_name = "text_similarity"
        self.model_type = "text_similarity"
        self.ph_output_type = "per_sequence"
        self.global_loss_buffer_size = global_loss_buffer_size
        self.generate_config()

    @classmethod
    def dot_product_scores(cls, query_vectors, passage_vectors):
        """
        Calculates dot product similarity scores for two 2-dimensional tensors

        :param query_vectors: tensor of query embeddings from BiAdaptive model
                        of dimension n1 x D,
                        where n1 is the number of queries/batch size and D is embedding size
        :type query_vectors: torch.Tensor
        :param passage_vectors: tensor of context/passage embeddings from BiAdaptive model
                        of dimension n2 x D,
                        where n2 is (batch_size * num_positives) + (batch_size * num_hard_negatives)
                        and D is embedding size
        :type passage_vectors: torch.Tensor

        :return dot_product: similarity score of each query with each context/passage (dimension: n1xn2)
        """
        # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
        dot_product = torch.matmul(query_vectors, torch.transpose(passage_vectors, 0, 1))
        return dot_product

    @classmethod
    def cosine_scores(cls, query_vectors, passage_vectors):
        """
        Calculates cosine similarity scores for two 2-dimensional tensors

        :param query_vectors: tensor of query embeddings from BiAdaptive model
                          of dimension n1 x D,
                          where n1 is the number of queries/batch size and D is embedding size
        :type query_vectors: torch.Tensor
        :param passage_vectors: tensor of context/passage embeddings from BiAdaptive model
                          of dimension n2 x D,
                          where n2 is (batch_size * num_positives) + (batch_size * num_hard_negatives)
                          and D is embedding size
        :type passage_vectors: torch.Tensor

        :return: cosine similarity score of each query with each context/passage (dimension: n1xn2)
        """
        # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
        cosine_similarities = []
        passages_per_batch = passage_vectors.shape[0]
        for query_vector in query_vectors:
            query_vector_repeated = query_vector.repeat(passages_per_batch, 1)
            current_cosine_similarities = nn.functional.cosine_similarity(query_vector_repeated, passage_vectors, dim=1)
            cosine_similarities.append(current_cosine_similarities)
        return torch.stack(cosine_similarities)

    def get_similarity_function(self):
        """
        Returns the type of similarity function used to compare queries and passages/contexts
        """
        if "dot_product" in self.similarity_function:
            return IANNHead.dot_product_scores
        elif "cosine" in self.similarity_function:
            return IANNHead.cosine_scores

    def forward(self, query_vectors: torch.Tensor, passage_vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Only packs the embeddings from both language models into a tuple. No further modification.
        The similarity calculation is handled later to enable distributed training (DDP)
        while keeping the support for in-batch negatives.
        (Gather all embeddings from nodes => then do similarity scores + loss)

        :param query_vectors: Tensor of query embeddings from BiAdaptive model
                          of dimension n1 x D,
                          where n1 is the number of queries/batch size and D is embedding size
        :type query_vectors: torch.Tensor
        :param passage_vectors: Tensor of context/passage embeddings from BiAdaptive model
                          of dimension n2 x D,
                          where n2 is the number of queries/batch size and D is embedding size
        :type passage_vectors: torch.Tensor

        :return: (query_vectors, passage_vectors)
        """
        return (query_vectors, passage_vectors)

    def _embeddings_to_scores(self, query_vectors: torch.Tensor, passage_vectors: torch.Tensor):
        """
        Calculates similarity scores between all given query_vectors and passage_vectors

        :param query_vectors: Tensor of queries encoded by the query encoder model
        :param passage_vectors: Tensor of passages encoded by the passage encoder model
        :return: Tensor of log softmax similarity scores of each query with each passage (dimension: n1xn2)
        """

        sim_func = self.get_similarity_function()
        scores = sim_func(query_vectors, passage_vectors)

        if len(query_vectors.size()) > 1:
            q_num = query_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = nn.functional.log_softmax(scores, dim=1)
        return softmax_scores

    def logits_to_loss(self, logits: Tuple[torch.Tensor, torch.Tensor], label_ids, **kwargs):
        """
        Computes the loss (Default: NLLLoss) by applying a similarity function (Default: dot product) to the input
        tuple of (query_vectors, passage_vectors) and afterwards applying the loss function on similarity scores.

        :param logits: Tuple of Tensors (query_embedding, passage_embedding) as returned from forward()

        :return: negative log likelihood loss from similarity scores
        """

        # Check if DDP is initialized
        # TODO: Remove it afterwards
        # try:
        #     rank = torch.distributed.get_rank()
        # except AssertionError:
        #     rank = -1
        rank = -1
        # Prepare predicted scores
        query_vectors, passage_vectors = logits

        # Prepare Labels
        positive_idx_per_question = torch.nonzero((label_ids.view(-1) == 1), as_tuple=False)

        # Gather global embeddings from all distributed nodes (DDP)
        if rank != -1:
            q_vector_to_send = torch.empty_like(query_vectors).cpu().copy_(query_vectors).detach_()
            p_vector_to_send = torch.empty_like(passage_vectors).cpu().copy_(passage_vectors).detach_()

            global_question_passage_vectors = all_gather_list(
                [q_vector_to_send, p_vector_to_send, positive_idx_per_question], max_size=self.global_loss_buffer_size
            )

            global_query_vectors = []
            global_passage_vectors = []
            global_positive_idx_per_question = []
            total_passages = 0
            for i, item in enumerate(global_question_passage_vectors):
                q_vector, p_vectors, positive_idx = item

                if i != rank:
                    global_query_vectors.append(q_vector.to(query_vectors.device))
                    global_passage_vectors.append(p_vectors.to(passage_vectors.device))
                    global_positive_idx_per_question.extend([v + total_passages for v in positive_idx])
                else:
                    global_query_vectors.append(query_vectors)
                    global_passage_vectors.append(passage_vectors)
                    global_positive_idx_per_question.extend([v + total_passages for v in positive_idx_per_question])
                total_passages += p_vectors.size(0)

            global_query_vectors = torch.cat(global_query_vectors, dim=0)
            global_passage_vectors = torch.cat(global_passage_vectors, dim=0)
            global_positive_idx_per_question = torch.LongTensor(global_positive_idx_per_question)
        else:
            global_query_vectors = query_vectors
            global_passage_vectors = passage_vectors
            global_positive_idx_per_question = positive_idx_per_question

        # Get similarity scores
        softmax_scores = self._embeddings_to_scores(global_query_vectors, global_passage_vectors)
        targets = global_positive_idx_per_question.squeeze(-1).to(softmax_scores.device)

        # Calculate loss
        loss = self.loss_fct(softmax_scores, targets)
        return loss

    def logits_to_preds(self, logits: Tuple[torch.Tensor, torch.Tensor], **kwargs):
        """
        Returns predicted ranks(similarity) of passages/context for each query

        :param logits: tensor of log softmax similarity scores of each query with each context/passage (dimension: n1xn2)
        :type logits: torch.Tensor

        :return: predicted ranks of passages for each query
        """
        query_vectors, passage_vectors = logits
        softmax_scores = self._embeddings_to_scores(query_vectors, passage_vectors)
        _, sorted_scores = torch.sort(softmax_scores, dim=1, descending=True)
        return sorted_scores

    def prepare_labels(self, label_ids, **kwargs):
        """
        Returns a tensor with passage labels(0:hard_negative/1:positive) for each query

        :return: passage labels(0:hard_negative/1:positive) for each query
        """
        labels = torch.zeros(label_ids.size(0), label_ids.numel())

        positive_indices = torch.nonzero(label_ids.view(-1) == 1, as_tuple=False)

        for i, indx in enumerate(positive_indices):
            labels[i, indx.item()] = 1
        return labels

    def formatted_preds(self, logits: Tuple[torch.Tensor, torch.Tensor], **kwargs):
        raise NotImplementedError("formatted_preds is not supported in TextSimilarityHead yet!")


__all__ = ["IRegressionHead", "ICLSHead", "LMHead", "NSPHead", "IANNHead", "IMLCLSHead", "ITOKENCLSHead"]
