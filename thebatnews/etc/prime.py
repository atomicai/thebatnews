import logging
import os
import pickle
import random
import signal
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import numpy as np
from torch import multiprocessing as mp
import torch
import torch.distributed as dist

from thebatnews.etc.visual import WORKER_F, WORKER_M, WORKER_X

logger = logging.getLogger(__name__)


class GracefulKiller:
    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True


def set_all_seeds(seed: int, deterministic_cudnn: bool = False) -> None:
    """
    Setting multiple seeds to make runs reproducible.

    Important: Enabling `deterministic_cudnn` gives you full reproducibility with CUDA,
    but might slow down your training (see https://pytorch.org/docs/stable/notes/randomness.html#cudnn) !

    :param seed:number to use as seed
    :param deterministic_cudnn: Enable for full reproducibility when using CUDA. Caution: might slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calc_chunksize(num_dicts, min_chunksize=4, max_chunksize=2000, max_processes=128):
    if mp.cpu_count() > 3:
        num_cpus = min(mp.cpu_count() - 1 or 1, max_processes)  # -1 to keep a CPU core free for xxx
    else:
        num_cpus = min(mp.cpu_count(), max_processes)  # when there are few cores, we use all of them

    dicts_per_cpu = np.ceil(num_dicts / num_cpus)
    # automatic adjustment of multiprocessing chunksize
    # for small files (containing few dicts) we want small chunksize to ulitize all available cores but never less
    # than 2, because we need it to sample another random sentence in LM finetuning
    # for large files we want to minimize processor spawning without giving too much data to one process, so we
    # clip it at 5k
    multiprocessing_chunk_size = int(np.clip((np.ceil(dicts_per_cpu / 5)), a_min=min_chunksize, a_max=max_chunksize))
    # This lets us avoid cases in lm_finetuning where a chunk only has a single doc and hence cannot pick
    # a valid next sentence substitute from another document
    if num_dicts != 1:
        while num_dicts % multiprocessing_chunk_size == 1:
            multiprocessing_chunk_size -= -1
    dict_batches_to_process = int(num_dicts / multiprocessing_chunk_size)
    num_processes = min(num_cpus, dict_batches_to_process) or 1

    return multiprocessing_chunk_size, num_processes


def initialize_device_settings(
    use_cuda: Optional[bool] = None,
    local_rank: int = -1,
    multi_gpu: bool = True,
    devices: Optional[List[Union[str, torch.device]]] = None,
) -> Tuple[List[torch.device], int]:
    """
    Returns a list of available devices.

    :param use_cuda: Whether to make use of CUDA GPUs (if available).
    :param local_rank: Ordinal of device to be used. If -1 and `multi_gpu` is True, all devices will be used.
                       Unused if `devices` is set or `use_cuda` is False.
    :param multi_gpu: Whether to make use of all GPUs (if available).
                      Unused if `devices` is set or `use_cuda` is False.
    :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
                        A list containing torch device objects and/or strings is supported (For example
                        [torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
                        parameter is not used and a single cpu device is used for inference.
    """
    if use_cuda is False:  # Note that it could be None, in which case we also want to just skip this step.
        devices_to_use = [torch.device("cpu")]
        n_gpu = 0
    elif devices:
        if not isinstance(devices, list):
            raise ValueError(f"devices must be a list, but got {devices} of type {type(devices)}")
        if any(isinstance(device, str) for device in devices):
            torch_devices: List[torch.device] = [torch.device(device) for device in devices]
            devices_to_use = torch_devices
        else:
            devices_to_use = devices  # type: ignore [assignment]
        n_gpu = sum(1 for device in devices_to_use if "cpu" not in device.type)
    elif local_rank == -1:
        if torch.cuda.is_available():
            if multi_gpu:
                devices_to_use = [torch.device(device) for device in range(torch.cuda.device_count())]
                n_gpu = torch.cuda.device_count()
            else:
                devices_to_use = [torch.device("cuda:0")]
                n_gpu = 1
        else:
            devices_to_use = [torch.device("cpu")]
            n_gpu = 0
    else:
        devices_to_use = [torch.device("cuda", local_rank)]
        torch.cuda.set_device(devices_to_use[0])
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")

    # HF transformers v4.21.2 pipeline object doesn't accept torch.device("cuda"), it has to be an indexed cuda device
    # TODO eventually remove once the limitation is fixed in HF transformers
    device_to_replace = torch.device("cuda")
    devices_to_use = [torch.device("cuda:0") if device == device_to_replace else device for device in devices_to_use]

    logger.info("Using devices: %s - Number of GPUs: %s", ", ".join([str(device) for device in devices_to_use]).upper(), n_gpu)
    return devices_to_use, n_gpu


def flatten_list(nested_list):
    """Flatten an arbitrarily nested list, without recursion (to avoid
    stack overflows). Returns a new list, the original list is unchanged.
    >> list(flatten_list([1, 2, 3, [4], [], [[[[[[[[[5]]]]]]]]]]))
    [1, 2, 3, 4, 5]
    >> list(flatten_list([[1, 2], 3]))
    [1, 2, 3]
    """
    nested_list = deepcopy(nested_list)

    while nested_list:
        sublist = nested_list.pop(0)

        if isinstance(sublist, list):
            nested_list = sublist + nested_list
        else:
            yield sublist


def log_ascii_workers(n, logger):
    m_worker_lines = WORKER_M.split("\n")
    f_worker_lines = WORKER_F.split("\n")
    x_worker_lines = WORKER_X.split("\n")
    all_worker_lines = []
    for _ in range(n):
        rand = np.random.randint(low=0, high=3)
        if rand % 3 == 0:
            all_worker_lines.append(f_worker_lines)
        elif rand % 3 == 1:
            all_worker_lines.append(m_worker_lines)
        else:
            all_worker_lines.append(x_worker_lines)
    zipped = zip(*all_worker_lines)
    for z in zipped:
        logger.info("  ".join(z))


# DDP utils
def all_reduce(tensor, group=None):
    if group is None:
        group = dist.group.WORLD
    return dist.all_reduce(tensor, group=group)


def all_gather_list(data, group=None, max_size=16384):
    """Gathers arbitrary data from all nodes into a list.
    Similar to :func:`~torch.distributed.all_gather` but for arbitrary Python
    data. Note that *data* must be picklable.
    Args:
        data (Any): data from the local worker to be gathered on other workers
        group (optional): group of the collective
    """
    SIZE_STORAGE_BYTES = 4  # int32 to encode the payload size

    enc = pickle.dumps(data)
    enc_size = len(enc)

    if enc_size + SIZE_STORAGE_BYTES > max_size:
        raise ValueError("encoded data exceeds max_size, this can be fixed by increasing buffer size: {}".format(enc_size))

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    buffer_size = max_size * world_size

    if not hasattr(all_gather_list, "_buffer") or all_gather_list._buffer.numel() < buffer_size:
        all_gather_list._buffer = torch.cuda.ByteTensor(buffer_size)
        all_gather_list._cpu_buffer = torch.ByteTensor(max_size).pin_memory()

    buffer = all_gather_list._buffer
    buffer.zero_()
    cpu_buffer = all_gather_list._cpu_buffer

    assert enc_size < 256**SIZE_STORAGE_BYTES, "Encoded object size should be less than {} bytes".format(
        256**SIZE_STORAGE_BYTES
    )

    size_bytes = enc_size.to_bytes(SIZE_STORAGE_BYTES, byteorder="big")

    cpu_buffer[0:SIZE_STORAGE_BYTES] = torch.ByteTensor(list(size_bytes))
    cpu_buffer[SIZE_STORAGE_BYTES : enc_size + SIZE_STORAGE_BYTES] = torch.ByteTensor(list(enc))

    start = rank * max_size
    size = enc_size + SIZE_STORAGE_BYTES
    buffer[start : start + size].copy_(cpu_buffer[:size])

    all_reduce(buffer, group=group)

    try:
        result = []
        for i in range(world_size):
            out_buffer = buffer[i * max_size : (i + 1) * max_size]
            size = int.from_bytes(out_buffer[0:SIZE_STORAGE_BYTES], byteorder="big")
            if size > 0:
                result.append(pickle.loads(bytes(out_buffer[SIZE_STORAGE_BYTES : size + SIZE_STORAGE_BYTES].tolist())))
        return result
    except pickle.UnpicklingError:
        raise Exception(
            "Unable to unpickle data from other workers. all_gather_list requires all "
            "workers to enter the function together, so this error usually indicates "
            "that the workers have fallen out of sync somehow. Workers can fall out of "
            "sync if one of them runs out of memory, or if there are other conditions "
            "in your training script that can cause one worker to finish an epoch "
            "while other workers are still iterating over their portions of the data."
        )


__all__ = [
    "GracefulKiller",
    "set_all_seeds",
    "calc_chunksize",
    "initialize_device_settings",
    "flatten_list",
    "log_ascii_workers",
    "all_gather_list",
]
