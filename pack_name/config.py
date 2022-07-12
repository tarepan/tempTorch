"""Whole Configuration"""


from typing import Optional
from dataclasses import dataclass

from omegaconf import MISSING

from .data.datamodule import ConfData
from .train import ConfTrain
from .model import ConfModel


CONF_DEFAULT_STR = """
seed: 1234
path_extend_conf: null
model:
    net:
        dim_i: 1
        child:
            dim_o: 1
            dropout: 0.5
    optim:
        learning_rate: 0.01
        sched_decay_rate: 0.01
        sched_decay_step: 1000
data:
    adress_data_root: ""
    corpus:
        train:
            name: ""
            download: False
        val:
            name: ""
            download: False
        test:
            name: ""
            download: False
        n_val: 10
        n_test: 10
    dataset:
        attr1: 1
        preprocess:
            attr2: 2
    loader:
        batch_size_train: 8
        batch_size_val: 1
        batch_size_test: 1
        num_workers: null
        pin_memory: null
train:

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
