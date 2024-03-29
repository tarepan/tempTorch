"""Whole Configuration"""


from dataclasses import dataclass

from omegaconf import MISSING
from configen import default, generate_conf_loader # pyright: ignore [reportMissingTypeStubs]
from lightlightning import ConfTrain               # pyright: ignore [reportMissingTypeStubs]

from .data.transform import ConfTransform
from .data.datamodule import ConfData
from .model import ConfModel


CONF_DEFAULT_STR = """
seed: 1234
path_extend_conf: null
transform:
    load:
        sampling_rate: 16000
    preprocess:
        raw2hoge:
            amp: 1.2
        raw2fuga:
            div: 3.0
    augment:
        len_clip: 10
model:
    net:
        feat_i: 1
        time_i: 1
        feat_o: 1
        child:
            dropout: 0.5
    optim:
        learning_rate: 0.01
        sched_decay_rate: 0.01
        sched_decay_step: 1000
    transform: "${transform}"
data:
    root: ""
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
        save_archive: False
        transform: "${transform}"
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
    """Configuration of everything."""
    seed:             int           = MISSING                  # PyTorch-Lightning's seed for every random system
    path_extend_conf: None | str    = MISSING                  # Path of configuration yaml which extends default config
    transform:        ConfTransform = default(ConfTransform())
    model:            ConfModel     = default(ConfModel())
    data:             ConfData      = default(ConfData())
    train:            ConfTrain     = default(ConfTrain())


# Exported
load_conf = generate_conf_loader(CONF_DEFAULT_STR, ConfGlobal)
"""Load configuration type-safely.
"""
