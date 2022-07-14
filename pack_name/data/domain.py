"""Data domain"""


from typing import Tuple

import numpy as np
from numpy.typing import NDArray


"""
(delele here when template is used)

[Design Notes - Separated domain]
    Data processing easily have circular dependencies.
    Internal data type of the data can be splitted into domain file.
"""


# Statically-preprocessed item
## Piyo :: (T,) - piyo piyo
Piyo = NDArray[np.float32]
## Hoge :: (T,) - hoge hoge
Hoge = NDArray[np.float32]
## Fuga :: (T,) - fuga fuga
Fuga = NDArray[np.float32]
## the item
HogeFuga = Tuple[Hoge, Fuga]

# Dynamically-transformed Dataset datum
## Hoge :: (T=t, 1) - hoge hoge
HogeDatum = NDArray[np.float32]
## Fuga :: (T=t, 1) - fuga fuga
FugaDatum = NDArray[np.float32]
## the datum
HogeFugaDatum = Tuple[Hoge, Fuga]
