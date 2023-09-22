import logging
import numbers

import numpy as np
import torch
from torch.utils.data import TensorDataset

from thebatnews.etc import flatten_list

logger = logging.getLogger(__name__)


def sample_to_features_text(sample, tasks, max_seq_len, tokenizer, process_fn):
    """
    Generates a dictionary of features for a given input sample that is to be consumed by a text classification model.

    :param sample: Sample object that contains human readable text and label fields from a single text classification data sample
    :type sample: Sample
    :param tasks: A dictionary where the keys are the names of the tasks and the values are the details of the task (e.g. label_list, metric, tensor name)
    :type tasks: dict
    :param max_seq_len: Sequences are truncated after this many tokens
    :type max_seq_len: int
    :param tokenizer: A tokenizer object that can turn string sentences into a list of tokens
    :return: A list with one dictionary containing the keys "input_ids", "padding_mask" and "segment_ids" (also "label_ids" if not
             in inference mode). The values are lists containing those features.
    :rtype: list
    """

    if tokenizer.is_fast:
        text = process_fn(sample.clear_text["text"])
        # Here, we tokenize the sample for the second time to get all relevant ids
        # This should change once we git rid of FARM's tokenize_with_metadata()
        inputs = tokenizer(
            text,
            return_token_type_ids=True,
            truncation=True,
            truncation_strategy="longest_first",
            max_length=max_seq_len,
            return_special_tokens_mask=True,
        )

        if (len(inputs["input_ids"]) - inputs["special_tokens_mask"].count(1)) != len(sample.tokenized["tokens"]):
            logger.error(
                "FastTokenizer encoded sample %s to %s tokens, which differs "
                "from number of tokens produced in tokenize_with_metadata(). \n"
                "Further processing is likely to be wrong.",
                sample.clear_text["text"],
                len(inputs["input_ids"]) - inputs["special_tokens_mask"].count(1),
            )
    else:
        # TODO It might be cleaner to adjust the data structure in sample.tokenized
        tokens_a = sample.tokenized["tokens"]
        tokens_b = sample.tokenized.get("tokens_b", None)

        inputs = tokenizer(
            tokens_a,
            tokens_b,
            add_special_tokens=True,
            truncation=False,  # truncation_strategy is deprecated
            return_token_type_ids=True,
            is_split_into_words=False,
        )

    input_ids, segment_ids = inputs["input_ids"], inputs["token_type_ids"]

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    padding_mask = [1] * len(input_ids)

    # Padding up to the sequence length.
    # Normal case: adding multiple 0 to the right
    # Special cases:
    # a) xlnet pads on the left and uses  "4"  for padding token_type_ids
    if tokenizer.__class__.__name__ == "XLNetTokenizer":
        pad_on_left = True
        segment_ids = np.pad(segment_ids, max_seq_len, 4, pad_on_left=pad_on_left)
    else:
        pad_on_left = False
        segment_ids = np.pad(segment_ids, max_seq_len, 0, pad_on_left=pad_on_left)

    input_ids = np.pad(input_ids, max_seq_len, tokenizer.pad_token_id, pad_on_left=pad_on_left)
    padding_mask = np.pad(padding_mask, max_seq_len, 0, pad_on_left=pad_on_left)

    assert len(input_ids) == max_seq_len
    assert len(padding_mask) == max_seq_len
    assert len(segment_ids) == max_seq_len

    feat_dict = {"input_ids": input_ids, "padding_mask": padding_mask, "segment_ids": segment_ids}

    # Add Labels for different tasks
    for task_name, task in tasks.items():
        try:
            label_name = task["label_name"]
            label_raw = sample.clear_text[label_name]
            label_list = task["label_list"]
            if task["task_type"] == "classification":
                # id of label
                try:
                    label_ids = [label_list.index(label_raw)]
                except ValueError:
                    raise ValueError(f"[Task: {task_name}] Observed label {label_raw} not in defined label_list")
            elif task["task_type"] == "multilabel_classification":
                # multi-hot-format
                label_ids = [0] * len(label_list)
                for l in label_raw.split(","):
                    if l != "":
                        label_ids[label_list.index(l)] = 1
            elif task["task_type"] == "regression":
                label_ids = [float(label_raw)]
            else:
                raise ValueError(task["task_type"])
        except KeyError:
            # For inference mode we don't expect labels
            label_ids = None
        if label_ids is not None:
            feat_dict[task["label_tensor_name"]] = label_ids
    return [feat_dict]


def convert_features_to_dataset(features):
    """
    Converts a list of feature dictionaries (one for each sample) into a PyTorch Dataset.

    :param features: A list of dictionaries. Each dictionary corresponds to one sample. Its keys are the
                     names of the type of feature and the keys are the features themselves.
    :Return: a Pytorch dataset and a list of tensor names.
    """
    # features can be an empty list in cases where down sampling occurs
    if len(features) == 0:
        return None, None
    tensor_names = list(features[0].keys())
    all_tensors = []
    for t_name in tensor_names:
        try:
            # Checking whether a non-integer will be silently converted to torch.long
            check = features[0][t_name]
            if isinstance(check, numbers.Number):
                base = check
            # extract a base variable from a nested lists or tuples
            elif isinstance(check, list):
                base = list(flatten_list(check))[0]
            # extract a base variable from numpy arrays
            else:
                base = check.ravel()[0]
            if not np.issubdtype(type(base), np.integer):
                logger.warning(
                    "Problem during conversion to torch tensors:\n"
                    "A non-integer value for feature '%s' with a value of: "
                    "'%s' will be converted to a torch tensor of dtype long.",
                    t_name,
                    base,
                )
        except:
            logger.debug("Could not determine type for feature '%s'. Converting now to a tensor of default type long.", t_name)

        # Convert all remaining python objects to torch long tensors
        cur_tensor = torch.as_tensor(np.array([sample[t_name] for sample in features]), dtype=torch.long)

        all_tensors.append(cur_tensor)

    dataset = TensorDataset(*all_tensors)
    return dataset, tensor_names


__all__ = ["sample_to_features_text", "convert_features_to_dataset"]
