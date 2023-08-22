"""The Network"""


from dataclasses import dataclass

from torch import nn, Tensor
from omegaconf import MISSING, II
from configen import default      # pyright: ignore [reportMissingTypeStubs]

from ..domain import HogeBatched
from .child import Child, ConfChild


@dataclass
class ConfNetwork:
    """Configuration of a `Network` instance."""
    feat_i: int      = MISSING            # Feature dimension size of input
    time_i: int      = MISSING            # Time    dimension size of input
    feat_o: int      = MISSING            # Feature dimension size of output
    child: ConfChild = default(ConfChild(
        feat_i=II("..feat_i"),
        feat_o=II("..feat_o"),))

class Network(nn.Module):
    """The Network.
    """
    def __init__(self, conf: ConfNetwork):
        super().__init__() # pyright: ignore [reportUnknownMemberType]

        # Submodule
        self._child = Child(conf.child)

    def forward(self, hoge: HogeBatched) -> Tensor: # pyright: ignore [reportIncompatibleMethodOverride]
        """(PT API) Forward a batch.

        Args:
            hoge                         - Input Hoge
        Returns:
            o_pred :: (B, T, Feat=dim_o) - Prediction
        """

        # :: (Batch, T, Feat=dim_i) -> (Batch, T, Feat=dim_o)
        o_pred = self._child(hoge)

        return o_pred

    def generate(self, hoge: HogeBatched) -> Tensor:
        """Run inference with a batch.

        Args:
            hoge                         - Input Hoge
        Returns:
            o_pred :: (B, T, Feat=dim_o) - Prediction
        """

        # :: (B, T, Feat=dim_i) -> (B, T, Feat=dim_o)
        o_pred = self._child(hoge)

        return o_pred
