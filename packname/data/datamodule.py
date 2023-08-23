"""Data wrapper by PL-datamodule"""


from dataclasses import dataclass

from torch.utils.data import DataLoader
from omegaconf import MISSING, SI
import lightning.pytorch as pl                                       # pyright: ignore [reportMissingTypeStubs]
from speechdatasety.helper.loader import generate_loader, ConfLoader # pyright: ignore [reportMissingTypeStubs]
from configen import default                                         # pyright: ignore [reportMissingTypeStubs]

from .domain import HogeFugaDatum
from .dataset import HogeFugaDataset, ConfHogeFugaDataset
from .corpus import prepare_corpora, ConfCorpora


@dataclass
class ConfData:
    """Configuration of the Data.
    """
    root:    str | None          = MISSING
    corpus:  ConfCorpora         = default(ConfCorpora(
        root=SI("${..root}")))
    dataset: ConfHogeFugaDataset = default(ConfHogeFugaDataset(
        root=SI("${..root}")))
    loader:  ConfLoader          = default(ConfLoader())

class Data(pl.LightningDataModule):
    """Data wrapper.
    """
    def __init__(self, conf: ConfData):
        super().__init__()
        self._conf = conf

    # def prepare_data(self) -> None:
    #     """(PL-API) Prepare data in dataset.
    #     """
    #     pass

    def setup(self, stage: str | None = None) -> None:
        """(PL-API) Setup train/val/test datasets.
        """

        corpus_train, corpus_val, corpus_test = prepare_corpora(self._conf.corpus)
        if stage == "fit" or stage is None:
            self.dataset_train = HogeFugaDataset(self._conf.dataset, *corpus_train) # pylint: disable=attribute-defined-outside-init
            self.dataset_val   = HogeFugaDataset(self._conf.dataset, *corpus_val)   # pylint: disable=attribute-defined-outside-init
        if stage == "test" or stage is None:
            self.dataset_test  = HogeFugaDataset(self._conf.dataset, *corpus_test)  # pylint: disable=attribute-defined-outside-init

    def train_dataloader(self) -> DataLoader[HogeFugaDatum]:
        """(PL-API) Generate training dataloader."""
        return generate_loader(self.dataset_train, self._conf.loader, "train")

    def val_dataloader(self) -> DataLoader[HogeFugaDatum]:
        """(PL-API) Generate validation dataloader."""
        return generate_loader(self.dataset_val,   self._conf.loader, "val")

    def test_dataloader(self) -> DataLoader[HogeFugaDatum]:
        """(PL-API) Generate test dataloader."""
        return generate_loader(self.dataset_test,  self._conf.loader, "test")
