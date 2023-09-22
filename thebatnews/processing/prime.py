import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import polars as pl
import simplejson as json
from transformers import AutoTokenizer

from thebatnews.processing.mask import IProcessor
from thebatnews.processing.feature import tokenize_with_metadata, truncate_sequences
from thebatnews.processing.sample import Sample, SampleBasket
from thebatnews.processing.tool import convert_features_to_dataset, sample_to_features_text

logger = logging.getLogger(__name__)


class ICLSProcessor(IProcessor):
    def __init__(
        self,
        tokenizer,
        max_seq_len,
        data_dir,
        label_list=None,
        metric=None,
        train_filename="train.csv",
        dev_filename=None,
        test_filename="test.csv",
        dev_split=0.1,
        dev_stratification=False,
        delimiter="\t",
        quote_char="'",
        skiprows=None,
        label_column_name="label",
        multilabel=False,
        header=0,
        proxies=None,
        max_samples=None,
        text_column_name="text",
        process_fn=None,
        **kwargs,
    ):
        # Custom processor attributes
        self.delimiter = delimiter
        self.quote_char = quote_char
        self.skiprows = skiprows
        self.header = header
        self.max_samples = max_samples
        self.dev_stratification = dev_stratification
        self.process_fn = process_fn

        logger.debug("Currently no support in Processor for returning problematic ids")

        super(ICLSProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            tasks={},
            proxies=proxies,
        )

        if metric and label_list:
            if multilabel:
                task_type = "multilabel_classification"
            else:
                task_type = "classification"
            self.add_task(
                name="text_classification",
                metric=metric,
                label_list=label_list,
                label_column_name=label_column_name,
                text_column_name=text_column_name,
                task_type=task_type,
            )
        else:
            logger.info(
                "Initialized processor without tasks. Supply `metric` and `label_list` to the constructor for "
                "using the default task or add a custom task later via processor.add_task()"
            )

    def file_to_dicts(self, file: str) -> List[Dict]:
        # arr = pl.read_csv(file).to_arrow()
        arr = pl.read_csv(file)
        response = []
        for task in self.tasks.values():
            x_col = task["text_column_name"]
            x_hat = "text"
            y_col = task["label_column_name"]
            y_hat = task["label_name"]  # How we want the label column to be named in out own flow.
            xs, ys = arr.select(x_col).to_series().to_list(), arr.select(y_col).to_series().to_list()
            xys = [{x_hat: x, y_hat: y} for x, y in zip(xs, ys)]
            response.extend(xys)
            # for xs, ys in zip(chunked(arr[x_col], n=10_000), chunked(arr[y_col], n=10_000)):
            #     xs = [str(x) for x in xs]
            #     ys = [str(y) for y in ys]
            #     batch = [{x_hat: x, y_hat: y} for x, y in zip(xs, ys)]
            #     response.extend(batch)
        return response

    def dataset_from_dicts(
        self, dicts: List[Dict], indices: Optional[List[int]] = None, return_baskets: bool = False, debug: bool = False
    ):
        if indices is None:
            indices = []
        baskets = []
        # Tokenize in batches
        texts = [self.process_fn(x["text"]) for x in dicts]
        tokenized_batch = self.tokenizer(
            texts,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
        )

        input_ids_batch = tokenized_batch["input_ids"]
        segment_ids_batch = tokenized_batch["token_type_ids"]
        padding_masks_batch = tokenized_batch["attention_mask"]
        tokens_batch = [x.tokens for x in tokenized_batch.encodings]

        for dictionary, input_ids, segment_ids, padding_mask, tokens in zip(
            dicts, input_ids_batch, segment_ids_batch, padding_masks_batch, tokens_batch
        ):
            tokenized = {}
            if debug:
                tokenized["tokens"] = tokens

            feat_dict = {"input_ids": input_ids, "padding_mask": padding_mask, "segment_ids": segment_ids}

            # Create labels
            # i.e. not inference
            if not return_baskets:
                label_dict = self.convert_labels(dictionary)
                feat_dict.update(label_dict)

            # Add Basket to baskets
            curr_sample = Sample(id="", clear_text=dictionary, tokenized=tokenized, features=[feat_dict])
            curr_basket = SampleBasket(id_internal=None, raw=dictionary, id_external=None, samples=[curr_sample])
            baskets.append(curr_basket)

        if indices and 0 not in indices:
            pass
        else:
            self._log_samples(n_samples=1, baskets=baskets)

        # TODO populate problematic ids
        problematic_ids: set = set()
        dataset, tensornames = self._create_dataset(baskets)
        if return_baskets:
            return dataset, tensornames, problematic_ids, baskets
        else:
            return dataset, tensornames, problematic_ids

    def convert_labels(self, dictionary: Dict):
        response: Dict = {}
        # Add labels for different tasks
        for task in self.tasks.values():
            label_name = task["label_name"]
            label_raw = dictionary[label_name]
            label_list = task["label_list"]
            if task["task_type"] == "classification":
                # id of label
                label_ids = [label_list.index(label_raw)]
            elif task["task_type"] == "multilabel_classification":
                # multi-hot-format
                label_ids = [0] * len(label_list)
                for l in label_raw.split(","):
                    if l != "":
                        label_ids[label_list.index(l)] = 1
            response[task["label_tensor_name"]] = label_ids
        return response

    def _create_dataset(self, baskets: List[SampleBasket]):
        features_flat: List = []
        basket_to_remove = []
        for basket in baskets:
            if self._check_sample_features(basket):
                if not isinstance(basket.samples, Iterable):
                    raise ValueError("basket.samples must contain a list of samples.")
                for sample in basket.samples:
                    if sample.features is None:
                        raise ValueError("sample.features must not be None.")
                    features_flat.extend(sample.features)
            else:
                # remove the entire basket
                basket_to_remove.append(basket)
        dataset, tensor_names = convert_features_to_dataset(features=features_flat)
        return dataset, tensor_names


class ICLSFastProcessor(ICLSProcessor):
    """
    Generic processor used at inference time:
    - fast
    - no labels
    - pure encoding of text into pytorch dataset
    - Doesn't read from file, but only consumes dictionaries (e.g. coming from API requests)
    """

    def __init__(self, tokenizer, max_seq_len, **kwargs):
        super(ICLSFastProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            train_filename=None,
            dev_filename=None,
            test_filename=None,
            dev_split=None,
            data_dir=None,
            tasks={},
        )

    @classmethod
    def load_from_dir(cls, load_dir: str):
        """
         Overwriting method from parent class to **always** load the InferenceProcessor instead of the specific class stored in the config.

        :param load_dir: str, directory that contains a 'processor_config.json'
        :return: An instance of an InferenceProcessor
        """
        # read config
        processor_config_file = Path(load_dir) / "processor_config.json"
        with open(processor_config_file) as f:
            config = json.load(f)
        # init tokenizer
        tokenizer = AutoTokenizer.from_pretrained(load_dir, tokenizer_class=config["tokenizer"])
        # we have to delete the tokenizer string from config, because we pass it as Object
        del config["tokenizer"]

        processor = cls.load(tokenizer=tokenizer, processor_name="ICLSFastProcessor", **config)
        for task_name, task in config["tasks"].items():
            processor.add_task(name=task_name, metric=task["metric"], label_list=task["label_list"])

        if processor is None:
            raise Exception

        return processor

    def file_to_dicts(self, file: str) -> List[Dict]:
        raise NotImplementedError

    def convert_labels(self, dictionary: Dict):
        # For inference we do not need labels
        ret: Dict = {}
        return ret

    # Private method to keep s3e pooling and embedding extraction working
    def _dict_to_samples(self, dictionary: Dict, **kwargs) -> Sample:
        # this tokenization also stores offsets
        tokenized = tokenize_with_metadata(self.process_fn(dictionary["text"]), self.tokenizer)
        # truncate tokens, offsets and start_of_word to max_seq_len that can be handled by the model
        truncated_tokens = {}
        for seq_name, tokens in tokenized.items():
            truncated_tokens[seq_name], _, _ = truncate_sequences(
                seq_a=tokens, seq_b=None, tokenizer=self.tokenizer, max_seq_len=self.max_seq_len
            )
        return Sample(id="", clear_text=dictionary, tokenized=truncated_tokens)

    # Private method to keep s3e pooling and embedding extraction working
    def _sample_to_features(self, sample: Sample) -> Dict:
        features = sample_to_features_text(
            sample=sample,
            tasks=self.tasks,
            max_seq_len=self.max_seq_len,
            tokenizer=self.tokenizer,
            process_fn=self.process_fn,
        )
        return features


class IANNProcessor(IProcessor):
    pass


__all__ = ["ICLSProcessor", "ICLSFastProcessor", "IANNProcessor"]
