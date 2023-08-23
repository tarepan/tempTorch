"""Data transformation"""


from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import MISSING
from torch import from_numpy, stack # pyright: ignore [reportUnknownVariableType] ; because of PyTorch ; pylint: disable=no-name-in-module
from configen import default        # pyright: ignore [reportMissingTypeStubs]

from ..domain import FugaBatched, HogeBatched, HogeFugaBatch, LenFuga
from .domain import FugaDatum, HogeDatum, HogeFuga, HogeFugaDatum, Raw, Hoge, Fuga


# [Data transformation]
#
#      load        preprocessing            augmentation              collation
#     -----> raw -----------------> item -----------------> datum -----------------> batch
#                 before training            in Dataset             in DataLoader

###################################################################################################################################
# [Load]
"""
(delele here when template is used)

[Design Notes - Load as transformation]
    Loading determines data shape, and load utilities frequently modify the data.
    In this meaning, load has similar 'transform' funtionality.
    For this reason, `load_raw` is placed here.
"""

@dataclass
class ConfLoad:
    """
    Configuration of piyo loading.
    Args:
        sampling_rate - Sampling rate
    """
    sampling_rate: int = MISSING

def load_raw(conf: ConfLoad, path: Path) -> Raw:
    """Load raw data 'piyo' from the adress."""

    # Audio Example (librosa is not handled by this template)
    import librosa # pyright: ignore [reportMissingImports, reportUnknownVariableType] ; pylint: disable=import-outside-toplevel,import-error
    raw: Raw = librosa.load(path, sr=conf.sampling_rate, mono=True)[0] # pyright: ignore [reportUnknownMemberType,reportUnknownVariableType]

    return raw # pyright: ignore [reportUnknownVariableType]

###################################################################################################################################
# [Preprocessing]

@dataclass
class ConfRaw2Hoge:
    """Configuration of `raw_to_hoge` preprocessing."""
    amp: float = MISSING # Amplification factor

def raw_to_hoge(conf: ConfRaw2Hoge, raw: Raw) -> Hoge:
    """Convert piyo to hoge."""
    # Amplification :: (T,) -> (T,)
    hoge: Hoge = raw * conf.amp

    return hoge


@dataclass
class ConfRaw2Fuga:
    """Configuration of `raw_to_fuga` preprocessing."""
    div: float = MISSING # Division factor

def raw_to_fuga(conf: ConfRaw2Fuga, raw: Raw) -> Fuga:
    """Convert piyo to fuga."""
    # Division :: (T,) -> (T,)
    fuga: Fuga = raw / conf.div

    return fuga

@dataclass
class ConfPreprocess:
    """Configuration of `preprocess` static transform.
    This parameters are used as a part of the dataset identifier.
    """
    raw2hoge: ConfRaw2Hoge = default(ConfRaw2Hoge())
    raw2fuga: ConfRaw2Fuga = default(ConfRaw2Fuga())

def preprocess(conf: ConfPreprocess, raw: Raw) -> HogeFuga:
    """Preprocessing (raw_to_item) - Statically process raw data into item."""
    return raw_to_hoge(conf.raw2hoge, raw), raw_to_fuga(conf.raw2fuga, raw)

###################################################################################################################################
# [Augmentation]

@dataclass
class ConfAugment:
    """
    Configuration of item-to-datum augmentation.
    Args:
        len_clip - Length of clipping
    """
    len_clip: int = MISSING

def augment(conf: ConfAugment, hoge_fuga: HogeFuga) -> HogeFugaDatum:
    """Augmentation (item_to_datum) - Dynamically modify item into datum.

    Clipping + DimensionExpansion
    """
    hoge, fuga = hoge_fuga

    # Clipping
    ## :: (T=t,) -> (T=L,)
    hoge = hoge[:conf.len_clip]
    ## :: (T=t,) -> (T=L,)
    fuga = fuga[:conf.len_clip]

    # Dimension expansion
    ## :: (T,) -> (T, 1)
    hoge_datum: HogeDatum = np.expand_dims(hoge, axis=-1) # pyright: ignore [reportUnknownMemberType] ; because of numpy
    ## :: (T,) -> (T, 1)
    fuga_datum: FugaDatum = np.expand_dims(fuga, axis=-1) # pyright: ignore [reportUnknownMemberType]; because of numpy

    return hoge_datum, fuga_datum

###################################################################################################################################
# [collation]

def collate(datums: list[HogeFugaDatum]) -> HogeFugaBatch:
    """Collation (datum_to_batch) - Bundle multiple datum into a batch."""

    hoge_batched: HogeBatched = stack([from_numpy(datum[0]) for datum in datums])
    fuga_batched: FugaBatched = stack([from_numpy(datum[1]) for datum in datums])
    len_fuga: LenFuga = [datum[1].shape[0] for datum in datums]

    return hoge_batched, fuga_batched, len_fuga

###################################################################################################################################

@dataclass
class ConfTransform:
    """Configuration of data transform."""
    load: ConfLoad = ConfLoad()
    preprocess: ConfPreprocess = ConfPreprocess()
    augment: ConfAugment = ConfAugment()
