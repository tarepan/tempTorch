"""Data domain"""


import numpy as np
from numpy.typing import NDArray


"""
(delele here when template is used)

[Design Notes - Separated domain]
    Data processing easily have circular dependencies.
    Internal data type of the data can be splitted into domain file.
"""

# Raw datum
Raw  = NDArray[np.float32] # :: (T,) - Raw datum

# Statically-preprocessed item
Hoge = NDArray[np.float32] # :: (T,) - hoge hoge
Fuga = NDArray[np.float32] # :: (T,) - fuga fuga
## the item
HogeFuga = tuple[Hoge, Fuga]
## Types
Hoge_: Hoge = np.array([1.], dtype=np.float32)
Fuga_: Fuga = np.array([1.], dtype=np.float32)
HogeFuga_: HogeFuga = (Hoge_, Fuga_)

# Dynamically-transformed Dataset datum
HogeDatum = NDArray[np.float32] # :: (T=t, 1) - hoge hoge
FugaDatum = NDArray[np.float32] # :: (T=t, 1) - fuga fuga
## the datum
HogeFugaDatum = tuple[Hoge, Fuga]
