"""Domain"""


from torch import Tensor # pyright: ignore [reportUnknownVariableType] ; because of PyTorch ; pylint: disable=no-name-in-module


"""
(delele here when template is used)

[Design Notes - Data type]
    Data is finally transformed by collate_fn in DataLoader, then consumed by x_step of the Model (Network consumes some of them).
    Both data-side and model-side depends on the data type.
    For this reason, the data type is separated as domain.
"""


# Data batch
HogeBatched = Tensor # :: (B=b, T=t, 1) - hoge hoge
FugaBatched = Tensor # :: (B=b, T=t, 1) - fuga fuga
LenFuga = list[int]  # :: (L=b,)        - Non-padded length of items in the FugaBatched
## the batch
HogeFugaBatch = tuple[HogeBatched, FugaBatched, LenFuga]
