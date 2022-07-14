"""Preprocessing"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import MISSING

from .domain import Piyo, Hoge, Fuga


def piyo_to_hoge(item: Piyo) -> Hoge:
    """Convert piyo to hoge.
    """
    hoge: Hoge = item
    return hoge


def piyo_to_fuga(item: Piyo) -> Fuga:
    """Convert piyo to fuga.
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

def preprocess_hogefuga(conf: ConfPreprocessing, processor: Any, path_piyo: Path, path_hoge: Path, path_fuga: Path) -> None:
    """Preprocess an item to hoge and fuga.

    Args:
        conf - Configuration
        processor - Processor instance, reused over items
        path_piyo - Path of input 'piyo'
        path_hoge - Path of output 'hoge'
        path_fuga - Path of output 'fuga'
    """
    # Load
    ## Audio
    # item: Piyo = librosa.load(path_piyo, sr=16000, mono=True)[0]
    item: Piyo = np.array([1. for _ in range(conf.attr2)], dtype=np.float32) # pyright: ignore [reportUnknownMemberType]

    # Transform
    hoge = piyo_to_hoge(item)
    fuga = piyo_to_fuga(item)

    # Save
    path_hoge.parent.mkdir(parents=True, exist_ok=True)
    path_fuga.parent.mkdir(parents=True, exist_ok=True)
    np.save(path_hoge, hoge) # pyright: ignore [reportUnknownMemberType]
    np.save(path_fuga, fuga) # pyright: ignore [reportUnknownMemberType]
