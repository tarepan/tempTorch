"""Whole Configuration"""


from typing import Optional
from dataclasses import dataclass

from omegaconf import MISSING
from configen import generate_conf_loader # pyright: ignore [reportMissingTypeStubs]
from lightlightning import ConfTrain # pyright: ignore [reportMissingTypeStubs]

from .data.datamodule import ConfData
from .model import ConfModel


CONF_DEFAULT_STR = """
seed: 1234
path_extend_conf: null
model:
    net:
        dim_i: 1
        dim_o: 1
        child:
            dropout: 0.5
    optim:
        learning_rate: 0.01
        sched_decay_rate: 0.01
        sched_decay_step: 1000
data:
    adress_data_root: ""
    corpus:
        train:
            name: "TEST"
            download: False
        val:
            name: "TEST"
            download: False
        test:
            name: "TEST"
            download: False
        n_val: 1
        n_test: 1
    dataset:
        attr1: 1
        preprocess:
            attr2: 100
    loader:
        batch_size_train: 1
        batch_size_val: 1
        batch_size_test: 1
        num_workers: null
        pin_memory: null
train:
    gradient_clipping: null
    max_epochs: 10000
    val_interval_epoch: 100
    profiler: null
    ckpt_log:
        dir_root: "."
        name_exp: "default"
        name_version: "version_0"
"""

@dataclass
class ConfGlobal:
    """Configuration of everything.
    Args:
        seed: PyTorch-Lightning's seed for every random system
        path_extend_conf: Path of configuration yaml which extends default config
    """
    seed: int = MISSING
    path_extend_conf: Optional[str] = MISSING
    model: ConfModel = ConfModel()
    data: ConfData = ConfData()
    train: ConfTrain = ConfTrain()


# Exported
load_conf = generate_conf_loader(CONF_DEFAULT_STR, ConfGlobal)
"""Load configuration type-safely.
"""
