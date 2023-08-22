"""The Child sub-module"""


from dataclasses import dataclass

from torch import nn, Tensor
from omegaconf import MISSING


@dataclass
class ConfChild:
    """Configuration of a `Child` instance."""
    feat_i:   int  = MISSING # Feature dimension size of input
    feat_o:   int  = MISSING # Feature dimension size of output
    dropout: float = MISSING # Dropout rate (0 means No-dropout)

class Child(nn.Module):
    """The Network.
    """
    def __init__(self, conf: ConfChild):
        super().__init__() # pyright: ignore [reportUnknownMemberType]

        layers: list[nn.Module] = []
        layers += [nn.Linear(conf.feat_i, conf.feat_o), nn.ReLU()]
        layers += [nn.Dropout(conf.dropout)] if conf.dropout > 0. else []
        self.fc1 = nn.Sequential(*layers)

    def forward(self, i_pred: Tensor) -> Tensor: # pyright: ignore [reportIncompatibleMethodOverride]
        """(PT API) Forward a batch.

        Arguments:
            i_pred :: (B, T, Feat=dim_i) - Input
        Returns:
            o_pred :: (B, T, Feat=dim_o) - Prediction
        """
        # :: (B, T, Feat=dim_i) -> (B, T, Feat=dim_o)
        o_pred = self.fc1(i_pred)

        return o_pred

    def generate(self, i_pred: Tensor) -> Tensor:
        """Run inference with a batch.

        Arguments:
            i_pred :: (B, T, Feat=dim_i) - Input
        Returns:
            o_pred :: (B, T, Feat=dim_o) - Prediction
        """

        # :: (B, T, Feat=dim_i) -> (B, T, Feat=dim_o)
        o_pred = self.fc1(i_pred)

        return o_pred
