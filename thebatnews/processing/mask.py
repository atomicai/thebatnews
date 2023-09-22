import abc
import inspect
import logging
import os
import random
from inspect import signature
from pathlib import Path
from typing import Dict, List, Optional, Union

import simplejson as json
from transformers import AutoTokenizer

from thebatnews.etc.format import is_json
from thebatnews.processing.sample import SampleBasket

logger = logging.getLogger(__name__)


class IProcessor(abc.ABC):
    """
    Base class for low level data processors to convert input text to PyTorch Datasets.
    """

    subclasses: dict = {}

    def __init__(
        self,
        tokenizer,
        max_seq_len: int,
        train_filename: Optional[Union[Path, str]],
        dev_filename: Optional[Union[Path, str]],
        test_filename: Optional[Union[Path, str]],
        dev_split: float,
        data_dir: Optional[Union[Path, str]],
        tasks: Optional[Dict] = None,
        proxies: Optional[Dict] = None,
        multithreading_rust: Optional[bool] = True,
        tracker=None,
    ):
        """
        :param tokenizer: Used to split a sentence (str) into tokens.
        :param max_seq_len: Samples are truncated after this many tokens.
        :param train_filename: The name of the file containing training data.
        :param dev_filename: The name of the file containing the dev data. If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :param test_filename: The name of the file containing test data.
        :param dev_split: The proportion of the train set that will be sliced. Only works if `dev_filename` is set to `None`.
        :param data_dir: The directory in which the train, test and perhaps dev files can be found.
        :param tasks: Tasks for which the processor shall extract labels from the input data.
                      Usually this includes a single, default task, e.g. text classification.
                      In a multitask setting this includes multiple tasks, e.g. 2x text classification.
                      The task name will be used to connect with the related PredictionHead.
        :param proxies: proxy configuration to allow downloads of remote datasets.
                    Format as in  "requests" library: https://2.python-requests.org//en/latest/user/advanced/#proxies
        :param multithreading_rust: Whether to allow multithreading in Rust, e.g. for FastTokenizers.
                                    Note: Enabling multithreading in Rust AND multiprocessing in python might cause
                                    deadlocks.
        """
        if tasks is None:
            tasks = {}
        if not multithreading_rust:
            os.environ["RAYON_RS_NUM_CPUS"] = "1"

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.tasks = tasks
        self.proxies = proxies

        # data sets
        self.train_filename = train_filename
        self.dev_filename = dev_filename
        self.test_filename = test_filename
        self.dev_split = dev_split
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = None  # type: ignore
        self.tracker = tracker

        self._log_params()
        self.problematic_sample_ids: set = set()

    def __init_subclass__(cls, **kwargs):
        """This automatically keeps track of all available subclasses.
        Enables generic load() and load_from_dir() for all specific Processor implementation.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    @classmethod
    def load(
        cls,
        processor_name: str,
        data_dir: str,  # TODO revert ignore
        tokenizer,  # type: ignore
        max_seq_len: int,
        train_filename: str,
        dev_filename: Optional[str],
        test_filename: str,
        dev_split: float,
        **kwargs,
    ):
        """
        Loads the class of processor specified by processor name.

        :param processor_name: The class of processor to be loaded.
        :param data_dir: Directory where data files are located.
        :param tokenizer: A tokenizer object
        :param max_seq_len: Sequences longer than this will be truncated.
        :param train_filename: The name of the file containing training data.
        :param dev_filename: The name of the file containing the dev data.
                             If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :param test_filename: The name of the file containing test data.
        :param dev_split: The proportion of the train set that will be sliced.
                          Only works if dev_filename is set to None
        :param kwargs: placeholder for passing generic parameters
        :return: An instance of the specified processor.
        """

        sig = signature(cls.subclasses[processor_name])
        unused_args = {k: v for k, v in kwargs.items() if k not in sig.parameters}
        logger.debug(
            "Got more parameters than needed for loading %s: %s. Those won't be used!", processor_name, unused_args
        )
        processor = cls.subclasses[processor_name](
            data_dir=data_dir,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            **kwargs,
        )

        return processor

    @classmethod
    def load_from_dir(cls, load_dir: str):
        """
         Infers the specific type of Processor from a config file (e.g. SquadProcessor) and loads an instance of it.

        :param load_dir: directory that contains a 'processor_config.json'
        :return: An instance of a Processor Subclass (e.g. SquadProcessor)
        """
        # read config
        processor_config_file = Path(load_dir) / "processor_config.json"
        with open(processor_config_file) as f:
            config = json.load(f)
        config["inference"] = True
        # init tokenizer
        if "lower_case" in config.keys():
            logger.warning(
                "Loading tokenizer from deprecated config. "
                "If you used `custom_vocab` or `never_split_chars`, this won't work anymore."
            )
            tokenizer = AutoTokenizer.from_pretrained(
                load_dir, tokenizer_class=config["tokenizer"], do_lower_case=config["lower_case"]
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(load_dir, tokenizer_class=config["tokenizer"])

        # we have to delete the tokenizer string from config, because we pass it as Object
        del config["tokenizer"]

        processor = cls.load(tokenizer=tokenizer, processor_name=config["processor"], **config)

        for task_name, task in config["tasks"].items():
            processor.add_task(
                name=task_name,
                metric=task["metric"],
                label_list=task["label_list"],
                label_column_name=task["label_column_name"],
                text_column_name=task.get("text_column_name", None),
                task_type=task["task_type"],
            )

        if processor is None:
            raise Exception

        return processor

    @classmethod
    def convert_from_transformers(
        cls,
        tokenizer_name_or_path,
        task_type,
        max_seq_len,
        doc_stride,
        revision=None,
        tokenizer_class=None,
        tokenizer_args=None,
        use_fast=True,
        max_query_length=64,
        **kwargs,
    ):
        tokenizer_args = tokenizer_args or {}
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            tokenizer_class=tokenizer_class,
            use_fast=use_fast,
            revision=revision,
            **tokenizer_args,
            **kwargs,
        )

        # TODO infer task_type automatically from config (if possible)
        if task_type == "embeddings":
            processor = InferenceProcessor(tokenizer=tokenizer, max_seq_len=max_seq_len)

        else:
            raise ValueError(
                f"`task_type` {task_type} is not supported yet. "
                "Valid options for arg `task_type`: 'question_answering', "
                "'embeddings', "
            )

        return processor

    def save(self, save_dir: str):
        """
        Saves the vocabulary to file and also creates a json file containing all the
        information needed to load the same processor.

        :param save_dir: Directory where the files are to be saved
        :return: None
        """
        os.makedirs(save_dir, exist_ok=True)
        config = self.generate_config()
        # save tokenizer incl. attributes
        config["tokenizer"] = self.tokenizer.__class__.__name__

        # Because the fast tokenizers expect a str and not Path
        # always convert Path to str here.
        self.tokenizer.save_pretrained(str(save_dir))

        # save processor
        config["processor"] = self.__class__.__name__
        output_config_file = Path(save_dir) / "processor_config.json"
        with open(output_config_file, "w") as file:
            json.dump(config, file)

    def generate_config(self):
        """
        Generates config file from Class and instance attributes (only for sensible config parameters).
        """
        config = {}
        # self.__dict__ doesn't give parent class attributes
        for key, value in inspect.getmembers(self):
            if is_json(value) and key[0] != "_":
                if issubclass(type(value), Path):
                    value = str(value)
                config[key] = value
        return config

    # TODO potentially remove tasks from code - multitask learning is not supported anyways
    def add_task(
        self, name, metric, label_list, label_column_name=None, label_name=None, task_type=None, text_column_name=None
    ):
        if type(label_list) is not list:
            raise ValueError(f"Argument `label_list` must be of type list. Got: f{type(label_list)}")

        if label_name is None:
            label_name = f"{name}_label"
        label_tensor_name = label_name + "_ids"
        self.tasks[name] = {
            "label_list": label_list,
            "metric": metric,
            "label_tensor_name": label_tensor_name,
            "label_name": label_name,
            "label_column_name": label_column_name,
            "text_column_name": text_column_name,
            "task_type": task_type,
        }

    @abc.abstractmethod
    def file_to_dicts(self, file: str) -> List[dict]:
        raise NotImplementedError()

    @abc.abstractmethod
    def dataset_from_dicts(
        self, dicts: List[Dict], indices: Optional[List[int]] = None, return_baskets: bool = False, debug: bool = False
    ):
        raise NotImplementedError()

    @abc.abstractmethod
    def _create_dataset(self, baskets: List[SampleBasket]):
        raise NotImplementedError

    @staticmethod
    def log_problematic(problematic_sample_ids):
        if problematic_sample_ids:
            n_problematic = len(problematic_sample_ids)
            problematic_id_str = ", ".join([str(i) for i in problematic_sample_ids])
            logger.error(
                "Unable to convert %s samples to features. Their ids are : %s", n_problematic, problematic_id_str
            )

    @staticmethod
    def _check_sample_features(basket: SampleBasket):
        """
        Check if all samples in the basket has computed its features.

        :param basket: the basket containing the samples

        :return: True if all the samples in the basket has computed its features, False otherwise
        """
        return basket.samples and not any(sample.features is None for sample in basket.samples)

    def _log_samples(self, n_samples: int, baskets: List[SampleBasket]):
        logger.debug("*** Show %s random examples ***", n_samples)
        if len(baskets) == 0:
            logger.debug("*** No samples to show because there are no baskets ***")
            return
        for _ in range(n_samples):
            random_basket = random.choice(baskets)
            random_sample = random.choice(random_basket.samples)  # type: ignore
            logger.debug(random_sample)

    def _log_params(self):
        params = {"processor": self.__class__.__name__, "tokenizer": self.tokenizer.__class__.__name__}
        names = ["max_seq_len", "dev_split"]
        for name in names:
            value = getattr(self, name)
            params.update({name: str(value)})
        if self.tracker:
            self.tracker.track_params(params)


__all__ = ["IProcessor"]
