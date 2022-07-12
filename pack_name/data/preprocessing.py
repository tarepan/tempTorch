"""Preprocessing"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import MISSING

from .dataset import Piyo, Hoge, Fuga


def item_to_hoge(item: Piyo) -> Hoge:
    """Convert item to hoge.
    """
    hoge: Hoge = item
    return hoge


def item_to_fuga(item: Piyo) -> Fuga:
    """Convert item to fuga.
    """
    fuga: Fuga = item
    return fuga


@dataclass
class ConfPreprocessing:
    """Configuration of preprocessing.
    Args:
        attr1 - Attribute #1
        attr2 - Attribute #2
    """
    attr1: int = MISSING
    attr2: int = MISSING

def preprocess_hogefuga(conf: ConfPreprocessing, processor: Any, path_item: Path, path_hoge: Path, path_fuga: Path) -> None:
    """Preprocess an item to hoge and fuga.

    Args:
        conf - Configuration
        processor - Processor instance, reused over items
        path_item - Path of input 'piyo'
        path_hoge - Path of output 'hoge'
        path_fuga - Path of output 'fuga'
    """
    # Load
    ## Audio
    # item: Piyo = librosa.load(path_item, sr=16000, mono=True)[0]
    item: Piyo = np.array([1.], dtype=np.float32)

    # Transform
    hoge = item_to_hoge(item)
    fuga = item_to_fuga(item)

    # Save
    np.save(path_hoge, hoge)
    np.save(path_fuga, fuga)
